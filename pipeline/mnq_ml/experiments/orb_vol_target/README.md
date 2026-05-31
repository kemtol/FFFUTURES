# MNQ ORB Volatility Targeted

Research-only program for New York opening range breakout on MNQ.

This experiment uses right-labeled M1 bars from the MNQ L0 pipeline:

```text
data/Level_0_Raw/MNQ_1m.duckdb
```

Daily confluence context is kept in a separate L0 DuckDB:

```text
data/Level_0_Raw/yfinance_daily.duckdb
table: daily_ohlcv
symbols: SPY, QQQ, VIX, TNX, DXY
```

No-lookahead contract: for MNQ trade date `D`, external daily confluence
features must only use rows with `date <= D-1`.

Outputs:

```text
data/Level_1_Features/mnq/orb_vol_target/context.parquet
data/Level_2_Datamart/mnq/orb_vol_target/events.parquet
data/Level_2_Datamart/mnq/orb_vol_target/sweeps/
model/MNQ/orb_vol_target/
```

Timing contract:

```text
M1 source: raw 1m bars shifted to right-label timestamps
Opening range candidates: 10m, 15m, 20m, 30m after 09:30 NY
Signal: first M1 close outside the selected opening range
Entry: next M1 open
Exit candidates: 15:00 NY time exit, or TP 2R before time exit
Sizing stop reference: opposite side of OR
```

## Current Candidate

Best current Topstep 30-day lens candidate:

```text
OR: 15m
Side: long only
Exit: TP 2R or 15:00 NY
Risk: $500
```

Evidence from the latest sweep:

| Window | Trades | PnL | Max DD |
| ---: | ---: | ---: | ---: |
| 5D | 1 | $101 | $0 |
| 10D | 5 | $895 | -$222 |
| 20D | 10 | $2,348 | -$222 |
| 30D | 18 | $3,460 | -$551 |
| 50D | 30 | $5,385 | -$861 |
| 100D | 54 | $4,035 | -$4,085 |
| 200D | 94 | $5,385 | -$4,561 |

Full sweep-row context for the current candidate:

| Metric | Value |
| --- | ---: |
| Full-history trades | 1,296 |
| Full-history win rate | 56.48% |
| Full-history PnL | $33,091 |
| Full-history max DD | -$12,124 |
| Full-history return/DD | 2.73 |
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
| 2026 trades | 72 |
| 2026 win rate | 59.72% |
| 2026 PnL | $6,096 |
| 2026 max DD | -$4,085 |
| Average contracts used | 3.48 |

Sharpe and Sortino are computed on daily dollar PnL over MNQ NY session days,
with zero PnL on no-trade days, annualized by `sqrt(252)`.

This is not live-ready. It is a research candidate that needs Topstep
consistency and MLL simulation before promotion.

Dedicated artifact folder:

```text
data/Level_2_Datamart/mnq/orb_vol_target/rule_based_15m_long_tp2r_eod/
```

## Rule-Based Baseline

Working name:

```text
MNQ ORB 15m Rule-Based Baseline
```

This strategy is intentionally non-ML. It is a deterministic rule-based
baseline that should remain easy to inspect, reproduce, compare, and later use
as the control group for ML filters or sizing overlays.

Daily procedure:

1. Use only the New York regular session.
2. Build the opening range from the first 15 M1 bars after 09:30 NY.
3. Mark the opening range high and low.
4. Wait for the first M1 candle after the range that closes above the OR high.
5. If that breakout occurs, enter long on the next M1 open.
6. Take at most one trade per NY trading day.
7. Size the trade from fixed dollar risk using OR low as the sizing reference.
8. Close early if price reaches +2R.
9. If +2R is not reached, close on the 15:00 NY time-exit bar.
10. Do not use ML predictions, indicator filters, discretionary overrides, or
    future bars in the baseline decision.

Research contract:

- Baseline side: long only.
- Baseline OR length: 15 minutes.
- Baseline risk: $500 target risk per trade.
- Baseline exit: TP 2R first, otherwise 15:00 NY time exit.
- Source grain: M1 right-labeled bars.
- Entry timing: signal at M1 close, execution at next M1 open.
- Promotion gate: Topstep-style profit target, MLL, consistency, daily PnL, and
  rolling-window robustness.

What ML may do later:

- Filter low-quality baseline trades.
- Reduce/increase risk allocation by regime.
- Compare against the rule-based baseline without changing the baseline itself.

What ML should not do first:

- Replace the entry logic before the deterministic edge is proven.
- Reduce the 30-day trade count so much that the strategy no longer fits the
  Topstep objective.

ML overlay structure:

```text
pipeline/mnq_ml/experiments/orb_vol_target/ml_filter/
data/Level_2_Datamart/mnq/orb_vol_target/ml_filter/
model/MNQ/orb_vol_target/ml_filter/
```

Current ML dataset:

| Field | Value |
| --- | ---: |
| Dataset | `candidate_a_dataset.parquet` |
| Rows | 1,296 |
| Columns | 27 |
| Train rows | 1,049 |
| Validation rows | 175 |
| Holdout rows | 72 |
| Holdout PnL baseline | $6,234 |
| Holdout win rate baseline | 59.72% |

Build command:

```bash
python3 pipeline/mnq_ml/experiments/orb_vol_target/ml_filter/build_ml_dataset.py --force
```

ORB risk-adjusted model structure:

```text
pipeline/mnq_ml/experiments/orb_vol_target/orb_risk_adjusted_model/
data/Level_2_Datamart/mnq/orb_vol_target/orb_risk_adjusted_model/
model/MNQ/orb_vol_target/orb_risk_adjusted_model/
```

This is the broader model objective:

- Estimate `P(up breakout reaches +2R)`.
- Estimate `P(down breakout reaches +2R)`.
- Treat no-breakout days as `NO_TRADE`.
- Do not model reversal yet.
- Action layer is only `FULL_RISK`, `REDUCE_RISK`, or `NO_TRADE`.

Current risk-adjusted dataset:

| Field | Value |
| --- | ---: |
| Dataset | `daily_scenarios.parquet` |
| Rows | 1,816 |
| Columns | 44 |
| Train rows | 1,457 |
| Validation rows | 257 |
| Holdout rows | 102 |
| First breakout UP | 927 |
| First breakout DOWN | 880 |
| No breakout | 9 |
| FULL_RISK labels | 322 |
| REDUCE_RISK labels | 1,485 |

Build command:

```bash
python3 pipeline/mnq_ml/experiments/orb_vol_target/orb_risk_adjusted_model/build_daily_scenarios.py --force
```

## Downside Extension Long Reversal

Working name:

```text
MNQ ORB Downside Extension Long Reversal
```

This is a separate long-only reversal experiment. It is not part of the current
continuation baseline.

Concept:

1. Build the 15m opening range.
2. If price extends below OR low by `X * OR range`, do not short.
3. Enter long on the next M1 open after the extension signal.
4. Use one OR range as trade risk by default.
5. SL: one OR range below entry.
6. TP: whichever is reached first, dynamic session VWAP or fixed R target.
7. Time exit: 15:00 NY.

Example:

```text
OR high = 1000
OR low = 800
OR range = 200
1R downside extension entry level = 600
2R downside extension entry level = 400
```

Current first-pass sweep:

```bash
python3 pipeline/mnq_ml/experiments/orb_vol_target/sweep_downside_extension_reversal.py --force
python3 pipeline/mnq_ml/experiments/orb_vol_target/sweep_downside_extension_reversal.py \
  --extension-r 1,1.5,2 \
  --rsi-period 14 \
  --rsi-max 30 \
  --output-dir data/Level_2_Datamart/mnq/orb_vol_target/downside_extension_reversal_rsi30 \
  --force
python3 pipeline/mnq_ml/experiments/orb_vol_target/sweep_downside_extension_reversal.py \
  --extension-r 1,1.5,2 \
  --tp-r 1 \
  --rsi-period 14 \
  --rsi-max 30 \
  --output-dir data/Level_2_Datamart/mnq/orb_vol_target/downside_extension_reversal_tp1_rsi30 \
  --force
```

Artifacts:

```text
data/Level_2_Datamart/mnq/orb_vol_target/downside_extension_reversal/
data/Level_2_Datamart/mnq/orb_vol_target/downside_extension_reversal_rsi30/
data/Level_2_Datamart/mnq/orb_vol_target/downside_extension_reversal_tp1/
data/Level_2_Datamart/mnq/orb_vol_target/downside_extension_reversal_tp1_rsi30/
```

Readout:

- The rule is mechanically possible.
- Full-history expectancy is negative across the first grid.
- RSI14 oversold gate is supported; first tested filter is `RSI14 <= 30` with
  extension `>= 1R`.
- RSI improves the deep `2R` branch's full-history damage but does not make it
  a clean positive full-history strategy.
- Recent results improve only at very deep `2R` downside extension, but trades
  are sparse and not enough for Topstep by itself.
- Changing fixed target from +2R to +1R improves win rate and reduces
  full-history drawdown, but it cuts the recent upside. For the most relevant
  branch, `2R` downside extension + `close_next_open` + `RSI14 <= 30` + $500
  risk, +1R gives full-history -$3,073 / DD -$6,385 and 50D +$1,172, while
  +2R gives full-history -$4,995 / DD -$10,124 and 50D +$1,990.
- Treat this as a research branch, not a replacement for the current long-only
  ORB continuation candidate.

## Artifacts

Current artifact map:

| Layer | Artifact | Rows | Columns | Purpose |
| --- | --- | ---: | ---: | --- |
| L0 | `data/Level_0_Raw/MNQ_1m.duckdb` | 2,487,265 | - | Raw MNQ M1 OHLCV source |
| L0 | `data/Level_0_Raw/yfinance_daily.duckdb` | 10,568 | 11 | Daily SPY/QQQ/VIX/TNX/DXY confluence source |
| L0 | `data/Level_0_Raw/yfinance_daily_manifest.json` | - | - | Daily confluence fetch summary and no-lookahead contract |
| L1 | `data/Level_1_Features/mnq/orb_vol_target/context.parquet` | 2,487,265 | 18 | Right-labeled M1 session context and OR fields |
| L1 | `data/Level_1_Features/mnq/orb_vol_target/context_manifest.json` | - | - | L1 build metadata |
| L1 | `data/Level_1_Features/mnq/orb_vol_target/l1_audit.json` | - | - | L1 integrity, continuity, null, and timing audit |
| L1 | `data/Level_1_Features/mnq/orb_vol_target/daily_confluence.parquet` | 2,199 | 31 | Prior daily SPY/QQQ/VIX/TNX/DXY confluence features; 29 feature columns |
| L1 | `data/Level_1_Features/mnq/orb_vol_target/daily_confluence_manifest.json` | - | - | Daily confluence feature manifest and no-lookahead contract |
| L1 | `data/Level_1_Features/mnq/orb_vol_target/daily_confluence_audit.json` | - | - | Daily confluence integrity and lookahead audit; latest status PASS |
| L2 | `data/Level_2_Datamart/mnq/orb_vol_target/events.parquet` | 637 | 27 | Canonical first M1 time-exit event file kept for reproducibility |
| L2 | `data/Level_2_Datamart/mnq/orb_vol_target/events_manifest.json` | - | - | Canonical event build metadata |
| L2 | `data/Level_2_Datamart/mnq/orb_vol_target/training_gate.json` | - | - | L1/L2 gate; latest status PASS |
| Sweep | `data/Level_2_Datamart/mnq/orb_vol_target/sweeps/sweep_base_opportunities.parquet` | 33,756 | 22 | Parameter-independent ORB opportunities |
| Sweep | `data/Level_2_Datamart/mnq/orb_vol_target/sweeps/sweep_events.parquet` | 159,294 | 28 | Executable events expanded by side, OR length, exit mode, and risk |
| Sweep | `data/Level_2_Datamart/mnq/orb_vol_target/sweeps/sweep_results.parquet` | 144 | 50 | Parameter-set ranking and PnL/DD metrics |
| Sweep | `data/Level_2_Datamart/mnq/orb_vol_target/sweeps/sweep_manifest.json` | - | - | Sweep grid and output metadata |
| ML | `data/Level_2_Datamart/mnq/orb_vol_target/ml_filter/candidate_a_dataset.parquet` | 1,296 | 27 | Model-ready Candidate A trade dataset |
| ML | `data/Level_2_Datamart/mnq/orb_vol_target/ml_filter/candidate_a_dataset_manifest.json` | - | - | ML dataset manifest and split summary |
| RiskAdjusted | `data/Level_2_Datamart/mnq/orb_vol_target/orb_risk_adjusted_model/daily_scenarios.parquet` | 1,816 | 44 | Daily up/down/no-breakout probability dataset |
| RiskAdjusted | `data/Level_2_Datamart/mnq/orb_vol_target/orb_risk_adjusted_model/daily_scenarios_manifest.json` | - | - | Risk-adjusted dataset manifest and label summary |
| RiskAdjusted | `data/Level_2_Datamart/mnq/orb_vol_target/orb_risk_adjusted_model/daily_scenarios_audit.json` | - | - | Risk-adjusted feature lookahead and integrity audit; latest status PASS |
| RiskAdjusted | `data/Level_2_Datamart/mnq/orb_vol_target/orb_risk_adjusted_model/breakout_quality_features.parquet` | 2,559 | 77 | Breakout-quality feature dataset with 62 features across volatility, VWAP, overnight, breakout, prior-day, and daily confluence families |
| RiskAdjusted | `data/Level_2_Datamart/mnq/orb_vol_target/orb_risk_adjusted_model/breakout_quality_features_manifest.json` | - | - | Breakout-quality feature family manifest |
| RiskAdjusted | `data/Level_2_Datamart/mnq/orb_vol_target/orb_risk_adjusted_model/breakout_quality_features_audit.json` | - | - | Breakout-quality no-lookahead and integrity audit; latest status PASS |
| Model | `model/MNQ/orb_vol_target/orb_risk_adjusted_model/risk_adjusted_v1_metrics.json` | - | - | V1 logistic + shallow LightGBM metrics for `success_2r` and `positive_eod` |
| Model | `model/MNQ/orb_vol_target/orb_risk_adjusted_model/risk_adjusted_v1_report.md` | - | - | V1 training summary |
| Model | `model/MNQ/orb_vol_target/orb_risk_adjusted_model/risk_adjusted_v1_*_*.joblib` | - | - | V1 trained probability model artifacts |
| Model | `model/MNQ/orb_vol_target/orb_risk_adjusted_model/risk_adjusted_v2_metrics.json` | - | - | V2 confluence logistic + shallow LightGBM metrics |
| Model | `model/MNQ/orb_vol_target/orb_risk_adjusted_model/risk_adjusted_v2_report.md` | - | - | V2 confluence training summary and feature-family readout |
| Model | `model/MNQ/orb_vol_target/orb_risk_adjusted_model/risk_adjusted_v2_*_*.joblib` | - | - | V2 trained probability model artifacts |
| RiskAdjusted | `data/Level_2_Datamart/mnq/orb_vol_target/orb_risk_adjusted_model/kelly_overlay_events.parquet` | 20,472 | 40 | Base-floor Kelly event rows with continuous and integer executable sizing |
| RiskAdjusted | `data/Level_2_Datamart/mnq/orb_vol_target/orb_risk_adjusted_model/kelly_overlay_summary.parquet` | - | - | Fixed vs base-floor Kelly overlay PnL/DD summary |
| RiskAdjusted | `data/Level_2_Datamart/mnq/orb_vol_target/orb_risk_adjusted_model/kelly_overlay_manifest.json` | - | - | Base-floor Kelly formula and parameters |
| Model | `model/MNQ/orb_vol_target/orb_risk_adjusted_model/risk_adjusted_v1_kelly_overlay_report.md` | - | - | Base-floor Kelly PnL/DD report with rounded-up integer contracts |
| RiskAdjusted | `data/Level_2_Datamart/mnq/orb_vol_target/orb_risk_adjusted_model/kelly_overlay_v2_events.parquet` | 20,472 | 40 | V2 confluence Kelly event rows with continuous and integer executable sizing |
| RiskAdjusted | `data/Level_2_Datamart/mnq/orb_vol_target/orb_risk_adjusted_model/kelly_overlay_v2_summary.parquet` | - | - | V2 fixed vs base-floor Kelly overlay PnL/DD summary |
| RiskAdjusted | `data/Level_2_Datamart/mnq/orb_vol_target/orb_risk_adjusted_model/kelly_overlay_v2_recent_windows.parquet` | 35 | 9 | V2 latest rolling-window integer PnL/DD snapshot |
| Model | `model/MNQ/orb_vol_target/orb_risk_adjusted_model/risk_adjusted_v2_kelly_overlay_report.md` | - | - | V2 base-floor Kelly PnL/DD report |
| Reversal | `data/Level_2_Datamart/mnq/orb_vol_target/downside_extension_reversal/downside_reversal_base_opportunities.parquet` | 4,604 | 27 | Downside extension long-reversal opportunities |
| Reversal | `data/Level_2_Datamart/mnq/orb_vol_target/downside_extension_reversal/downside_reversal_events.parquet` | 23,240 | 33 | Downside extension reversal events expanded by risk |
| Reversal | `data/Level_2_Datamart/mnq/orb_vol_target/downside_extension_reversal/downside_reversal_results.parquet` | 48 | 57 | Downside extension reversal sweep results |
| Reversal | `data/Level_2_Datamart/mnq/orb_vol_target/downside_extension_reversal/README.md` | - | - | Downside extension reversal first-pass report |
| Reversal | `data/Level_2_Datamart/mnq/orb_vol_target/downside_extension_reversal_rsi30/downside_reversal_results.parquet` | 36 | 59 | Downside extension reversal sweep with `RSI14 <= 30`, extension `>= 1R` |

Latest data/audit snapshot:

| Check | Value |
| --- | ---: |
| L1 min timestamp | 2019-05-05 22:04 UTC |
| L1 max timestamp | 2026-05-28 01:53 UTC |
| NY days | 2,199 |
| OR-complete days | 1,815 |
| Duplicate timestamps | 0 |
| Bad OHLC rows | 0 |
| L1 audit status | PASS |
| Training gate status | PASS |

Latest sweep snapshot:

| Field | Value |
| --- | ---: |
| Sweep anchor | 2026-05-28 01:53 UTC |
| OR minutes tested | 10, 15, 20, 30 |
| Sides tested | long, short, long_short |
| Exit modes tested | time_exit, tp_2r_or_time |
| Risk values tested | $100, $200, $300, $400, $500, $600 |
| Result rows | 144 |

## Main Commands

```bash
python3 pipeline/mnq_ml/experiments/orb_vol_target/build_l1_context.py --force
python3 pipeline/mnq_ml/experiments/orb_vol_target/build_orb_events.py --force
python3 pipeline/mnq_ml/experiments/orb_vol_target/audit_l1_context.py
python3 pipeline/mnq_ml/experiments/orb_vol_target/gate_training_data.py
python3 pipeline/mnq_ml/experiments/orb_vol_target/sweep_orb_params.py --force
python3 pipeline/mnq_ml/fetch_yfinance_daily_confluence.py
python3 pipeline/mnq_ml/experiments/orb_vol_target/build_daily_confluence_features.py --force
python3 pipeline/mnq_ml/experiments/orb_vol_target/audit_daily_confluence_features.py
python3 pipeline/mnq_ml/experiments/orb_vol_target/orb_risk_adjusted_model/train_risk_adjusted_v2.py --force
python3 pipeline/mnq_ml/experiments/orb_vol_target/orb_risk_adjusted_model/simulate_kelly_overlay_v2.py --force
python3 pipeline/mnq_ml/experiments/orb_vol_target/sweep_downside_extension_reversal.py --force
```

The canonical baseline event file remains deterministic. Parameter selection
comes from the sweep artifacts under:

```text
data/Level_2_Datamart/mnq/orb_vol_target/sweeps/
```

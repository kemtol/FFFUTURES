# MNQ ORB Risk-Adjusted Models

Model artifacts for the ORB risk-adjusted probability engine.

Goal:

```text
P(up breakout reaches +2R)
P(down breakout reaches +2R)
NO_TRADE if no breakout occurs
```

The first live action layer should only choose:

```text
FULL_RISK / REDUCE_RISK / NO_TRADE
```

No reversal model belongs here until the simple risk-adjusted model is validated.

## V1 Artifacts

First training iteration. These artifacts were trained before daily confluence
was added, so they are retained only as a pre-confluence baseline:

```text
risk_adjusted_v1_success_2r_logistic.joblib
risk_adjusted_v1_success_2r_lgbm_shallow.joblib
risk_adjusted_v1_positive_eod_logistic.joblib
risk_adjusted_v1_positive_eod_lgbm_shallow.joblib
risk_adjusted_v1_metrics.json
risk_adjusted_v1_feature_importance.csv
risk_adjusted_v1_report.md
risk_adjusted_v1_kelly_overlay_report.md
```

## V2 Artifacts

Confluence training iteration:

```text
risk_adjusted_v2_success_2r_logistic.joblib
risk_adjusted_v2_success_2r_lgbm_shallow.joblib
risk_adjusted_v2_positive_eod_logistic.joblib
risk_adjusted_v2_positive_eod_lgbm_shallow.joblib
risk_adjusted_v2_metrics.json
risk_adjusted_v2_feature_importance.csv
risk_adjusted_v2_report.md
risk_adjusted_v2_kelly_overlay_report.md
```

Dataset:

```text
data/Level_2_Datamart/mnq/ORB/orb_risk_adjusted_model/breakout_quality_features.parquet
```

Current dataset rows: 2,559. Columns: 77. Features: 62. Holdout rows: 140.

Current feature families:

| Family | Count |
| --- | ---: |
| Volatility / ATR | 9 |
| VWAP context | 4 |
| Overnight structure | 5 |
| Breakout quality | 11 |
| Prior-day context | 4 |
| Daily confluence | 29 |

Daily confluence uses `SPY`, `QQQ`, `VIX`, `TNX`, and `DXY` from
`data/Level_0_Raw/yfinance_daily.duckdb`. For MNQ trade date `D`, all daily
confluence features use external data with `date < D`.

## Feature Dictionary

| Family | Feature | Meaning |
| --- | --- | --- |
| Volatility / ATR | `prior_day_range_pts` | Prior MNQ regular-session high-low range in points. |
| Volatility / ATR | `atr5_daily_range_pts` | 5-day average of prior MNQ daily ranges, shifted before the trade day. |
| Volatility / ATR | `atr14_daily_range_pts` | 14-day average of prior MNQ daily ranges, shifted before the trade day. |
| Volatility / ATR | `atr20_daily_range_pts` | 20-day average of prior MNQ daily ranges, shifted before the trade day. |
| Volatility / ATR | `orb_range_to_atr14` | Opening-range size divided by 14-day daily range average. |
| Volatility / ATR | `signal_risk_to_atr14` | Breakout risk to the opposite OR boundary divided by 14-day daily range average. |
| Volatility / ATR | `pre_60m_range_pts` | MNQ range during the 60 minutes before NY open. |
| Volatility / ATR | `pre_60m_return_pts` | MNQ return from 60 minutes before NY open into the open. |
| Volatility / ATR | `pre_60m_realized_vol_pts` | Standard deviation of MNQ 1m close changes in the 60 minutes before NY open. |
| VWAP context | `or_close_dist_to_or_vwap_pts` | OR closing price minus VWAP of the opening range. |
| VWAP context | `signal_close_dist_to_vwap_pts` | Breakout close minus session VWAP up to the breakout candle. |
| VWAP context | `vwap_slope_or_to_signal_pts` | Change from OR VWAP to breakout-time VWAP. |
| VWAP context | `side_aligned_with_vwap` | 1 when breakout close is on the side-favorable side of VWAP. |
| Overnight structure | `overnight_range_pts` | Overnight high-low range before NY regular session. |
| Overnight structure | `overnight_return_pts` | Overnight return from first available overnight bar into NY open. |
| Overnight structure | `or_high_to_overnight_high_pts` | Overnight high minus OR high. Positive means upside liquidity remains above OR. |
| Overnight structure | `or_low_to_overnight_low_pts` | OR low minus overnight low. Positive means downside liquidity remains below OR. |
| Overnight structure | `side_distance_to_overnight_extreme_pts` | Distance from breakout close to the side-relevant overnight extreme. |
| Breakout quality | `side_is_up` | 1 for UP breakout, 0 for DOWN breakout. |
| Breakout quality | `signal_minutes_from_open` | Minutes from NY open to the breakout signal candle. |
| Breakout quality | `breakout_close_distance_pts` | Breakout close distance beyond OR boundary in points. |
| Breakout quality | `breakout_close_distance_to_orb` | Breakout close distance beyond OR boundary divided by OR range. |
| Breakout quality | `breakout_body_pts` | Absolute candle body size of the breakout candle. |
| Breakout quality | `breakout_range_pts` | High-low range of the breakout candle. |
| Breakout quality | `breakout_close_position` | Close location inside breakout candle, side-adjusted so higher is stronger. |
| Breakout quality | `breakout_wick_against_pts` | Wick length against the breakout direction. |
| Breakout quality | `breakout_volume` | Volume of the breakout candle. |
| Breakout quality | `breakout_volume_to_or_mean` | Breakout volume divided by average OR candle volume. |
| Breakout quality | `breakout_volume_to_pre60_mean` | Breakout volume divided by average pre-open 60m volume. |
| Prior-day context | `prior_day_trend_pts` | Prior regular-session close minus open in MNQ points. |
| Prior-day context | `prior_close_gap_pts` | Current OR first open minus prior regular-session close. |
| Prior-day context | `or_mid_to_prior_close_pts` | OR midpoint minus prior regular-session close. |
| Prior-day context | `side_distance_to_prior_extreme_pts` | Distance from breakout close to side-relevant prior-day high or low. |
| Daily confluence | `dc_spy_return_1d` | SPY one-day return through the prior external daily close. |
| Daily confluence | `dc_spy_return_5d` | SPY five-trading-day return through `D-1`. |
| Daily confluence | `dc_spy_return_20d` | SPY twenty-trading-day return through `D-1`. |
| Daily confluence | `dc_spy_dist_sma20` | SPY distance from its 20D moving average through `D-1`. |
| Daily confluence | `dc_spy_dist_sma50` | SPY distance from its 50D moving average through `D-1`. |
| Daily confluence | `dc_spy_realized_vol_20d` | SPY 20D daily return volatility through `D-1`. |
| Daily confluence | `dc_qqq_return_1d` | QQQ one-day return through `D-1`. |
| Daily confluence | `dc_qqq_return_5d` | QQQ five-trading-day return through `D-1`. |
| Daily confluence | `dc_qqq_return_20d` | QQQ twenty-trading-day return through `D-1`. |
| Daily confluence | `dc_qqq_dist_sma20` | QQQ distance from its 20D moving average through `D-1`. |
| Daily confluence | `dc_qqq_dist_sma50` | QQQ distance from its 50D moving average through `D-1`. |
| Daily confluence | `dc_qqq_realized_vol_20d` | QQQ 20D daily return volatility through `D-1`. |
| Daily confluence | `dc_qqq_minus_spy_return_1d` | QQQ 1D return minus SPY 1D return, a one-day Nasdaq relative-strength measure. |
| Daily confluence | `dc_qqq_minus_spy_return_5d` | QQQ 5D return minus SPY 5D return, short-term Nasdaq leadership. |
| Daily confluence | `dc_qqq_minus_spy_return_20d` | QQQ 20D return minus SPY 20D return, monthly Nasdaq leadership. |
| Daily confluence | `dc_qqq_spy_beta_60d` | Rolling 60D beta of QQQ daily returns to SPY daily returns. |
| Daily confluence | `dc_qqq_beta_residual_5d` | QQQ 5D return minus beta-adjusted SPY 5D return. |
| Daily confluence | `dc_qqq_beta_residual_20d` | QQQ 20D return minus beta-adjusted SPY 20D return. |
| Daily confluence | `dc_vix_prev_close` | VIX prior daily close, the volatility/fear level before the trade day. |
| Daily confluence | `dc_vix_change_1d` | VIX one-day point change through `D-1`. |
| Daily confluence | `dc_vix_change_5d` | VIX five-day point change through `D-1`. |
| Daily confluence | `dc_vix_percentile_20d` | VIX percentile versus its trailing 20D window through `D-1`. |
| Daily confluence | `dc_vix_percentile_60d` | VIX percentile versus its trailing 60D window through `D-1`. |
| Daily confluence | `dc_vix_dist_sma20` | VIX distance from its 20D moving average through `D-1`. |
| Daily confluence | `dc_tnx_prev_close` | 10Y yield prior daily close. |
| Daily confluence | `dc_tnx_change_1d` | 10Y yield one-day point change through `D-1`. |
| Daily confluence | `dc_tnx_change_5d` | 10Y yield five-day point change through `D-1`. |
| Daily confluence | `dc_dxy_return_1d` | DXY one-day return through `D-1`. |
| Daily confluence | `dc_dxy_return_5d` | DXY five-trading-day return through `D-1`. |

## V1 Readout

These V1 metrics were trained before the daily confluence family was added.
V2 below is the active confluence benchmark.

| Target | Model | Validation AUC | Holdout AUC | Holdout Brier |
| --- | --- | ---: | ---: | ---: |
| `success_2r` | logistic | 0.722 | 0.716 | 0.197 |
| `success_2r` | lgbm_shallow | 0.623 | 0.606 | 0.140 |
| `positive_eod` | logistic | 0.534 | 0.542 | 0.249 |
| `positive_eod` | lgbm_shallow | 0.547 | 0.550 | 0.242 |

Interpretation:

- `success_2r` has real first-pass separation, especially logistic.
- `positive_eod` is weak in V1 and should not drive sizing alone yet.
- LightGBM is better calibrated for `success_2r` but ranks weaker than logistic.
- This is research only and now superseded by V2 for confluence comparisons.

## V2 Readout

V2 uses the same model families as V1, trained on the 62-feature dataset.

| Target | Model | Validation AUC | Holdout AUC | Holdout PR-AUC | Holdout Brier | Delta vs V1 AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `success_2r` | logistic | 0.739 | 0.716 | 0.364 | 0.212 | -0.001 |
| `success_2r` | lgbm_shallow | 0.690 | 0.633 | 0.236 | 0.138 | +0.027 |
| `positive_eod` | logistic | 0.551 | 0.584 | 0.641 | 0.246 | +0.042 |
| `positive_eod` | lgbm_shallow | 0.576 | 0.515 | 0.588 | 0.245 | -0.034 |

Readout:

- `success_2r` logistic remains the best ranker, but confluence does not materially improve its holdout AUC.
- `success_2r` LightGBM improves versus V1 and has the best Brier score, but ranking is still weaker than logistic.
- `positive_eod` logistic improves, useful as diagnostic, but it is not the primary sizing target.
- Daily confluence is meaningful in feature importance, especially for logistic, but it is not a clean standalone breakthrough yet.

## Base-Floor Kelly Overlay V2

Risk-sizing simulation using V2 `success_2r` logistic probabilities. Every
breakout is still traded; ML only changes desired `risk_usd`.

Desired risk is floored at `$500`; Kelly only adds risk above that baseline.

The report now includes two views:

- Continuous research sizing: allows fractional contracts.
- Integer executable sizing: `max(1, ceil(risk_usd / risk_per_contract_usd))`.

Artifacts:

```text
data/Level_2_Datamart/mnq/ORB/orb_risk_adjusted_model/kelly_overlay_events.parquet
data/Level_2_Datamart/mnq/ORB/orb_risk_adjusted_model/kelly_overlay_summary.parquet
data/Level_2_Datamart/mnq/ORB/orb_risk_adjusted_model/kelly_overlay_manifest.json
data/Level_2_Datamart/mnq/ORB/orb_risk_adjusted_model/kelly_overlay_v2_events.parquet
data/Level_2_Datamart/mnq/ORB/orb_risk_adjusted_model/kelly_overlay_v2_summary.parquet
data/Level_2_Datamart/mnq/ORB/orb_risk_adjusted_model/kelly_overlay_v2_manifest.json
```

Holdout, integer executable view with rounded-up contracts:

| Variant | PnL | Max DD | Return/DD | Avg desired risk | Avg actual risk |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fixed_1.00x` | $17,140 | -$3,430 | 5.00 | $500 | $652 |
| `basefloor_kelly_0.10x` | $18,076 | -$3,436 | 5.26 | $511 | $669 |
| `basefloor_kelly_0.25x` | $17,827 | -$4,238 | 4.21 | $528 | $678 |
| `basefloor_kelly_0.50x` | $18,523 | -$4,238 | 4.37 | $557 | $698 |
| `basefloor_kelly_1.00x` | $22,006 | -$4,954 | 4.44 | $614 | $756 |
| `norm_target_600` | $21,676 | -$4,954 | 4.38 | $600 | $746 |
| `norm_target_750` | $27,857 | -$6,220 | 4.48 | $732 | $883 |
| `norm_target_1000` | $26,460 | -$6,258 | 4.23 | $818 | $969 |

Recent-window integer executable view, anchored to the latest available MNQ date:

| Variant | 5D PnL/DD | 10D PnL/DD | 20D PnL/DD | 30D PnL/DD |
| --- | ---: | ---: | ---: | ---: |
| `fixed_1.00x` | -$243 / -$2,142 | -$887 / -$2,142 | $2,296 / -$2,142 | $1,423 / -$3,430 |
| `basefloor_kelly_0.10x` | -$243 / -$2,142 | -$887 / -$2,142 | $2,558 / -$2,142 | $1,680 / -$3,436 |
| `basefloor_kelly_1.00x` | -$243 / -$2,142 | -$2,009 / -$2,445 | $3,151 / -$2,445 | $1,794 / -$3,914 |
| `norm_target_600` | -$243 / -$2,142 | -$2,009 / -$2,445 | $2,869 / -$2,445 | $1,410 / -$4,017 |
| `norm_target_750` | -$526 / -$2,916 | -$2,970 / -$3,124 | $3,802 / -$3,124 | $2,778 / -$4,034 |

Readout:

- Base-floor Kelly uses `clip(1 + kelly_fraction * raw_kelly, 1.0, 2.0)`.
- Normalized Kelly fits the scale on train only, then applies it to validation
  and holdout.
- Rounded-up contracts make actual risk higher than desired risk.
- Holdout improves meaningfully for normalized V2 overlays, but the latest 5D
  and 10D windows are negative.
- The risk-adjusted model is still not a live candidate; next gate is
  probability calibration and a Topstep MLL/consistency simulator, then HMM or
  other regime features only if the failure mode is clear.

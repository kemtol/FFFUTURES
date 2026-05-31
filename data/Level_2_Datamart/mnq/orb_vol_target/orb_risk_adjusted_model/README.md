# MNQ ORB Risk-Adjusted Model Datamart

Daily risk-adjusted datasets for the MNQ ORB probability model.

Current expected artifacts:

| File | Purpose |
| --- | --- |
| `daily_scenarios.parquet` | One row per NY day after 15m OR completion |
| `daily_scenarios_manifest.json` | Dataset summary, feature list, label list, and no-lookahead note |
| `daily_scenarios_audit.json` | Daily scenario integrity and no-lookahead audit |
| `breakout_quality_features.parquet` | One row per actual UP/DOWN breakout with enriched feature families |
| `breakout_quality_features_manifest.json` | Breakout-quality feature family and label manifest |
| `breakout_quality_features_audit.json` | Breakout-quality integrity and no-lookahead audit |
| `kelly_overlay_events.parquet` | Continuous and integer executable base-floor Kelly event simulation |
| `kelly_overlay_summary.parquet` | PnL/DD summary by split and Kelly variant |
| `kelly_overlay_manifest.json` | Kelly formula, parameters, and summary metadata |
| `kelly_overlay_v2_events.parquet` | V2 confluence probability Kelly event simulation |
| `kelly_overlay_v2_summary.parquet` | V2 confluence Kelly PnL/DD summary |
| `kelly_overlay_v2_manifest.json` | V2 Kelly formula, parameters, and summary metadata |
| `kelly_overlay_v2_recent_windows.parquet` | Latest 5D/10D/20D/30D/50D/100D/200D V2 integer-window summary |

This datamart is separate from `ml_filter/`:

- `ml_filter/`: narrow Candidate A trade-quality overlay.
- `orb_risk_adjusted_model/`: broader daily probability model for long/short breakout
  continuation and no-breakout no-trade days.

Current quick-win feature families in `breakout_quality_features.parquet`:

- `volatility_atr`
- `vwap_context`
- `overnight_structure`
- `breakout_quality`
- `prior_day_context`
- `daily_confluence`

Latest breakout-quality shape:

| Metric | Value |
| --- | ---: |
| Rows | 2,559 |
| Columns | 77 |
| Feature columns | 62 |
| Daily confluence features | 29 |

The daily confluence family is sourced from
`data/Level_0_Raw/yfinance_daily.duckdb` and includes prior daily SPY, QQQ,
VIX, TNX, and DXY context. It is joined by `ny_date` with the strict
no-lookahead contract `daily_confluence_feature_date < ny_date`.

Breakout-quality labels:

- `success_2r`: reaches +2R before 15:00 NY.
- `positive_eod`: exits with positive PnL, either TP or time exit.
- `outcome_bucket`: `TP_2R`, `POSITIVE_EOD`, or `NEGATIVE_EOD`.
- `r_multiple`: continuous R outcome.

Latest breakout-quality audit status is PASS. The audit verifies feature/label
nulls, forbidden label-like feature names, timestamp order, split leakage,
rebuild consistency, label definitions, and a high-correlation leakage smoke
check.

Base-floor Kelly overlay V2:

- Uses V2 `success_2r` logistic probabilities.
- Keeps every valid breakout.
- Desired risk is floored at $500.
- Base-floor formula: `risk_multiplier = clip(1 + kelly_fraction * raw_kelly, 1.0, 2.0)`.
- Normalized variants fit scale on train only for target average risks $600, $750, and $1000.
- Integer executable sizing is included with `contracts_minrisk_ceil = max(1, ceil(risk_usd / risk_per_contract_usd))`.
- Holdout improves for several V2 overlays, especially normalized $600/$750.
- Latest 5D/10D windows are negative, so this is not live-ready.

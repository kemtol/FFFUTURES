# MNQ ORB Risk-Adjusted Model

This is the broader probability model for the MNQ ORB program.

It is intentionally simpler than a reversal framework:

- If an upside breakout occurs, estimate probability of +2R continuation.
- If a downside breakout occurs, estimate probability of +2R continuation.
- If no breakout occurs before the cutoff, label the day as `NO_TRADE`.
- No reversal trade is modeled yet.

The action layer is deliberately narrow:

```text
FULL_RISK or REDUCE_RISK for long/short breakouts
NO_TRADE for no-breakout days
```

## Dataset

One row is one NY trading day after the 15m opening range is complete.

Output:

```text
data/Level_2_Datamart/mnq/orb_vol_target/orb_risk_adjusted_model/daily_scenarios.parquet
data/Level_2_Datamart/mnq/orb_vol_target/orb_risk_adjusted_model/daily_scenarios_manifest.json
```

Command:

```bash
python3 pipeline/mnq_ml/experiments/orb_vol_target/orb_risk_adjusted_model/build_daily_scenarios.py --force
python3 pipeline/mnq_ml/experiments/orb_vol_target/orb_risk_adjusted_model/audit_daily_scenarios.py
python3 pipeline/mnq_ml/fetch_yfinance_daily_confluence.py
python3 pipeline/mnq_ml/experiments/orb_vol_target/build_daily_confluence_features.py --force
python3 pipeline/mnq_ml/experiments/orb_vol_target/audit_daily_confluence_features.py
python3 pipeline/mnq_ml/experiments/orb_vol_target/orb_risk_adjusted_model/build_breakout_quality_features.py --force
python3 pipeline/mnq_ml/experiments/orb_vol_target/orb_risk_adjusted_model/audit_breakout_quality_features.py
python3 pipeline/mnq_ml/experiments/orb_vol_target/orb_risk_adjusted_model/train_risk_adjusted_v1.py --force
python3 pipeline/mnq_ml/experiments/orb_vol_target/orb_risk_adjusted_model/simulate_kelly_overlay.py --force
python3 pipeline/mnq_ml/experiments/orb_vol_target/orb_risk_adjusted_model/train_risk_adjusted_v2.py --force
python3 pipeline/mnq_ml/experiments/orb_vol_target/orb_risk_adjusted_model/simulate_kelly_overlay_v2.py --force
```

Latest build snapshot:

| Metric | Value |
| --- | ---: |
| Rows | 1,816 |
| Columns | 44 |
| Train rows | 1,457 |
| Validation rows | 257 |
| Holdout rows | 102 |
| First breakout UP | 927 |
| First breakout DOWN | 880 |
| No breakout | 9 |
| First-breakout FULL_RISK labels | 322 |
| First-breakout REDUCE_RISK labels | 1,485 |
| Upside breakout days | 1,322 |
| Upside +2R success rate | 13.07% |
| Upside avg PnL/contract | $10.16 |
| Downside breakout days | 1,261 |
| Downside +2R success rate | 20.78% |
| Downside avg PnL/contract | $1.17 |

Latest audit snapshot:

| Check | Result |
| --- | --- |
| Status | PASS |
| Feature window | pre-60m through OR completion only |
| OR feature minutes | `1..15` after 09:30 NY |
| Pre-feature minutes | `-60..0` before/at 09:30 NY |
| Label-like feature names | none |
| Feature nulls | none |
| Recomputed feature mismatches | none |
| Scenario coverage | 1,816 / 1,816 OR-complete dates |
| Label null consistency | PASS |
| Signal/entry/exit timestamp order | PASS |
| Split leakage | none |

Audit output:

```text
data/Level_2_Datamart/mnq/orb_vol_target/orb_risk_adjusted_model/daily_scenarios_audit.json
```

## Breakout Quality Features

This is the quick-win feature enrichment for the first trainable probability
model. One row is one actual UP or DOWN breakout. Features are available at
breakout candle close; entry/exit/PnL are labels or metadata only.

Output:

```text
data/Level_2_Datamart/mnq/orb_vol_target/orb_risk_adjusted_model/breakout_quality_features.parquet
data/Level_2_Datamart/mnq/orb_vol_target/orb_risk_adjusted_model/breakout_quality_features_manifest.json
data/Level_2_Datamart/mnq/orb_vol_target/orb_risk_adjusted_model/breakout_quality_features_audit.json
```

Feature families:

| Family | Purpose |
| --- | --- |
| `volatility_atr` | Whether +2R is realistic for the day's range regime |
| `vwap_context` | Whether breakout is aligned with intraday value |
| `overnight_structure` | Whether breakout is extended into overnight levels |
| `breakout_quality` | Breakout candle strength, timing, volume, and extension |
| `prior_day_context` | Position versus prior day range and close |
| `daily_confluence` | Prior daily SPY/QQQ/VIX/TNX/DXY regime and relative-strength context |

Latest breakout-quality build:

| Metric | Value |
| --- | ---: |
| Rows | 2,559 |
| Columns | 77 |
| Feature columns | 62 |
| Train rows | 2,068 |
| Validation rows | 351 |
| Holdout rows | 140 |
| UP rows | 1,308 |
| DOWN rows | 1,251 |
| Overall +2R rate | 16.84% |
| UP +2R rate | 13.07% |
| DOWN +2R rate | 20.78% |
| Overall positive EOD rate | 52.48% |
| UP positive EOD rate | 56.96% |
| DOWN positive EOD rate | 47.80% |
| TP 2R outcomes | 431 |
| Positive EOD outcomes | 912 |
| Negative EOD outcomes | 1,216 |

Latest breakout-quality audit:

| Check | Result |
| --- | --- |
| Status | PASS |
| Feature cutoff | breakout candle close |
| Daily confluence cutoff | prior external daily close only: `daily_confluence_feature_date < ny_date` |
| Entry open as feature | disabled |
| Entry/exit/PnL fields | labels or metadata only |
| Feature nulls | none |
| Label nulls | none |
| Forbidden feature names | none |
| Label leakage correlation smoke | PASS |
| `positive_eod` definition | `pnl_per_contract_usd > 0` |
| `outcome_bucket` definition | `TP_2R`, `POSITIVE_EOD`, `NEGATIVE_EOD` |
| Signal/entry/exit timestamp order | PASS |
| Rebuild consistency | PASS |

## Training V1

Command:

```bash
python3 pipeline/mnq_ml/experiments/orb_vol_target/orb_risk_adjusted_model/train_risk_adjusted_v1.py --force
```

Artifacts:

```text
model/MNQ/orb_vol_target/orb_risk_adjusted_model/risk_adjusted_v1_metrics.json
model/MNQ/orb_vol_target/orb_risk_adjusted_model/risk_adjusted_v1_report.md
model/MNQ/orb_vol_target/orb_risk_adjusted_model/risk_adjusted_v1_feature_importance.csv
model/MNQ/orb_vol_target/orb_risk_adjusted_model/risk_adjusted_v1_*_*.joblib
```

V1 result:

| Target | Best first-pass signal | Holdout readout |
| --- | --- | --- |
| `success_2r` | logistic ranking | AUC 0.716, but probability calibration still needs threshold/overlay work |
| `positive_eod` | weak | AUC about 0.55, not strong enough to drive sizing alone |

This is not live approval. V1 only proves whether the current feature families
can separate good breakout quality from weak breakout quality.

## Training V2

V2 trains the same model families on the 62-feature confluence dataset.

Command:

```bash
python3 pipeline/mnq_ml/experiments/orb_vol_target/orb_risk_adjusted_model/train_risk_adjusted_v2.py --force
```

Artifacts:

```text
model/MNQ/orb_vol_target/orb_risk_adjusted_model/risk_adjusted_v2_metrics.json
model/MNQ/orb_vol_target/orb_risk_adjusted_model/risk_adjusted_v2_report.md
model/MNQ/orb_vol_target/orb_risk_adjusted_model/risk_adjusted_v2_feature_importance.csv
model/MNQ/orb_vol_target/orb_risk_adjusted_model/risk_adjusted_v2_*_*.joblib
```

V2 result:

| Target | Model | Holdout AUC | Holdout PR-AUC | Holdout Brier | Delta AUC vs V1 |
| --- | --- | ---: | ---: | ---: | ---: |
| `success_2r` | logistic | 0.716 | 0.364 | 0.212 | -0.001 |
| `success_2r` | lgbm_shallow | 0.633 | 0.236 | 0.138 | +0.027 |
| `positive_eod` | logistic | 0.584 | 0.641 | 0.246 | +0.042 |
| `positive_eod` | lgbm_shallow | 0.515 | 0.588 | 0.245 | -0.034 |

Readout:

- Daily confluence is useful in feature importance, but it does not create a
  clean logistic `success_2r` ranking improvement.
- `success_2r` logistic remains the main probability source for Kelly because
  it still ranks best.
- V2 is a benchmark before HMM, not a live approval.

## Base-Floor Kelly Overlay V2

The current Kelly overlay uses the `success_2r` logistic model and keeps every
valid breakout. It changes only desired bet size. Desired risk is floored at
the base $500 risk; Kelly only adds size when probability is strong.

Executable MNQ sizing uses integer contracts rounded up:

```text
contracts_minrisk_ceil = max(1, ceil(risk_usd / risk_per_contract_usd))
```

Contract:

```text
base_risk_usd = 500
payoff_ratio = 2.0
min_risk_multiplier = 1.00
max_risk_multiplier = 2.00
kelly_fractions = 0.10, 0.25, 0.50, 1.00
normalized_target_risks = 600, 750, 1000
```

Base-floor fractional Kelly formula:

```text
risk_multiplier = clip(1 + kelly_fraction * max(0, (b*p - (1-p))/b), min=1.0, max=2.0)
pnl_usd = r_multiple * risk_usd
```

Base-floor normalized Kelly formula:

```text
scale is fit on train only
risk_multiplier = clip(1 + scale * max(0, (b*p - (1-p))/b), min=1.0, max=2.0)
```

V1 holdout result, integer executable view with rounded-up contracts:

| Variant | Rows | Int PnL | Int Max DD | Int Return/DD | Avg desired risk | Avg actual risk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_1.00x` | 140 | $17,140 | -$3,430 | 5.00 | $500 | $652 |
| `basefloor_kelly_0.10x` | 140 | $17,838 | -$3,436 | 5.19 | $510 | $665 |
| `basefloor_kelly_0.25x` | 140 | $18,518 | -$3,910 | 4.74 | $524 | $679 |
| `basefloor_kelly_0.50x` | 140 | $17,863 | -$4,238 | 4.22 | $548 | $693 |
| `basefloor_kelly_1.00x` | 140 | $19,836 | -$5,355 | 3.70 | $595 | $746 |
| `norm_target_600` | 140 | $17,494 | -$5,246 | 3.33 | $582 | $729 |
| `norm_target_750` | 140 | $23,738 | -$6,327 | 3.75 | $694 | $847 |
| `norm_target_1000` | 140 | $25,182 | -$6,934 | 3.63 | $793 | $945 |

Readout:

- Base-floor Kelly now respects the minimum desired $500 risk.
- Integer rounding-up makes actual average risk higher than desired risk.
- `basefloor_kelly_0.10x` is the cleanest first improvement over fixed in
  holdout return/DD; stronger Kelly fractions raise PnL but also raise DD.
- Current ML value is diagnostic: it detects breakout quality, but the first
  production candidate still needs Topstep-style MLL and consistency gates.

V2 holdout result, integer executable view with rounded-up contracts:

| Variant | Rows | Int PnL | Int Max DD | Int Return/DD | Avg desired risk | Avg actual risk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_1.00x` | 140 | $17,140 | -$3,430 | 5.00 | $500 | $652 |
| `basefloor_kelly_0.10x` | 140 | $18,076 | -$3,436 | 5.26 | $511 | $669 |
| `basefloor_kelly_0.25x` | 140 | $17,827 | -$4,238 | 4.21 | $528 | $678 |
| `basefloor_kelly_0.50x` | 140 | $18,523 | -$4,238 | 4.37 | $557 | $698 |
| `basefloor_kelly_1.00x` | 140 | $22,006 | -$4,954 | 4.44 | $614 | $756 |
| `norm_target_600` | 140 | $21,676 | -$4,954 | 4.38 | $600 | $746 |
| `norm_target_750` | 140 | $27,857 | -$6,220 | 4.48 | $732 | $883 |
| `norm_target_1000` | 140 | $26,460 | -$6,258 | 4.23 | $818 | $969 |

Latest V2 recent-window readout is weaker than holdout:

| Variant | 5D PnL/DD | 10D PnL/DD | 20D PnL/DD | 30D PnL/DD |
| --- | ---: | ---: | ---: | ---: |
| `fixed_1.00x` | -$243 / -$2,142 | -$887 / -$2,142 | $2,296 / -$2,142 | $1,423 / -$3,430 |
| `basefloor_kelly_0.10x` | -$243 / -$2,142 | -$887 / -$2,142 | $2,558 / -$2,142 | $1,680 / -$3,436 |
| `norm_target_750` | -$526 / -$2,916 | -$2,970 / -$3,124 | $3,802 / -$3,124 | $2,778 / -$4,034 |

Current P0 decision:

- Confluence V2 is worth keeping as the new benchmark.
- It is not enough for live forward testing.
- Before HMM, add Topstep MLL/consistency and calibration gates so the next
  feature family addresses a measured failure mode.

## Labels

| Label | Meaning |
| --- | --- |
| `up_success_2r` | Upside breakout reached +2R before 15:00 NY |
| `down_success_2r` | Downside breakout reached +2R before 15:00 NY |
| `first_breakout_side` | First side to close outside OR: `UP`, `DOWN`, or `NONE` |
| `first_breakout_risk_label` | `FULL_RISK`, `REDUCE_RISK`, or `NO_TRADE` |
| `success_2r` | Breakout-quality row reached +2R before 15:00 NY |
| `positive_eod` | Breakout-quality row finished positive at TP or time exit |
| `outcome_bucket` | `TP_2R`, `POSITIVE_EOD`, or `NEGATIVE_EOD` |
| `r_multiple` | Continuous R outcome for sizing/evaluation |

`FULL_RISK` currently means the first breakout reached +2R in hindsight.
Training should convert this into a probability threshold, not hard-code the
hindsight label into live decisions.

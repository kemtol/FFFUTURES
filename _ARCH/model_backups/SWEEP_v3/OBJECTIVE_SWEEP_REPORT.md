# Objective Sweep Report v3 — Scale-Invariant Features

Generated: 2026-04-26 22:35

## Key Changes from v2

| Aspect | v2 (baseline) | v3 (scale-invariant) |
|--------|:-------------:|:--------------------:|
| Features | 11 original | **18 (11 + 7 scale-invariant)** |
| New features | — | `breakout_strength_atr_ratio`, `atr14_sq`, `breakout_strength_sq`, `price_vs_vwap_pct_abs`, `orb_range_sq`, `adx_50_flag`, `breakout_strength_vs_orb` |
| AB test | — | Both models trained, results compared |
| Training data | 2010-2023 (same) | **Same** for fair comparison |
| Holdout | 2025+ (same) | **Same** |

## Why Scale-Invariant Features?

The 2026 collapse diagnosis found root cause: **model OOD failure due to volatility explosion**:

| Metric | Training (2010-2021) | 2024 | 2025 | 2026 |
|--------|:-------------------:|:----:|:----:|:----:|
| ATR14 median | $0.76 | $0.73 | $1.57 | **$3.53** |
| ORB range median | $3.50 | $3.50 | $7.30 | **$16.40** |
| Ratio vs training | 1.0× | 1.0× | 2.1× | **4.6×** |

Raw features like `breakout_strength` and `atr14_at_entry` see values 4-5× their training range.
Scale-invariant features normalize these by current ATR, so the model sees consistent distributions.

## Summary

- Labels tested: 10
- Holdout: 2025-01-01 onward
- Topstep account: 50K (target $3,000, MLL $2,000)
- Scoring: `v3_score = pass_rate - fail_mll_rate` (best params from 128-point grid)
- Best label (v3): `y_1r4_180m` (v3_score=+0.223)
- Labels with v3_score > 0: 8

## AB Test: v3 (All Features) vs v2 (Baseline Only)

| Metric | Value |
|--------|:-----:|
| Targets improved (v3 > v2) | 8/10 |
| Targets degraded (v3 < v2) | 2/10 |
| Best improvement | y_1r4_120m (+0.122) |
| Worst degradation | y_1r2_180m (-0.066) |

| target | v3_score | v2_score | delta | improved |
|---|---|---|---|---|
| y_1r4_180m | +0.223 | +0.188 | +0.035 | ✅ |
| y_1r4_120m | +0.198 | +0.075 | +0.122 | ✅ |
| y_1r4_60m | +0.166 | +0.147 | +0.019 | ✅ |
| y_1r4_240m | +0.163 | +0.141 | +0.022 | ✅ |
| y_1r4_close60m | +0.119 | +0.069 | +0.050 | ✅ |
| y_1r2_120m | +0.097 | +0.013 | +0.085 | ✅ |
| y_1r2_60m | +0.056 | +0.031 | +0.025 | ✅ |
| y_1r2_180m | +0.013 | +0.078 | -0.066 | ❌ |
| y_1r2_close60m | -0.035 | -0.016 | -0.019 | ❌ |
| y_1r2_240m | -0.053 | -0.072 | +0.019 | ✅ |

## Ranked Results (v3 — all features, best params per label)

| target | rr | wr_rev_baseline | wr_cont_baseline | v3_score | v3_pass_rate | v3_fail_mll | v3_median_pnl | v2_score | v2_pass_rate | v2_fail_mll | delta_score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| y_1r4_180m | 4.000 | 0.177 | 0.206 | 0.223 | 0.260 | 0.038 | 399.000 | 0.188 | 0.229 | 0.041 | 0.035 |
| y_1r4_120m | 4.000 | 0.171 | 0.198 | 0.198 | 0.282 | 0.085 | 1074.000 | 0.075 | 0.179 | 0.103 | 0.122 |
| y_1r4_60m | 4.000 | 0.143 | 0.176 | 0.166 | 0.166 | 0.000 | 430.000 | 0.147 | 0.248 | 0.100 | 0.019 |
| y_1r4_240m | 4.000 | 0.180 | 0.208 | 0.163 | 0.175 | 0.013 | 395.000 | 0.141 | 0.185 | 0.044 | 0.022 |
| y_1r4_close60m | 4.000 | 0.151 | 0.180 | 0.119 | 0.166 | 0.047 | -160.500 | 0.069 | 0.163 | 0.094 | 0.050 |
| y_1r2_120m | 2.000 | 0.322 | 0.350 | 0.097 | 0.188 | 0.091 | 805.000 | 0.013 | 0.129 | 0.116 | 0.085 |
| y_1r2_60m | 2.000 | 0.318 | 0.344 | 0.056 | 0.329 | 0.273 | 1038.000 | 0.031 | 0.075 | 0.044 | 0.025 |
| y_1r2_180m | 2.000 | 0.323 | 0.350 | 0.013 | 0.226 | 0.213 | 1104.000 | 0.078 | 0.198 | 0.119 | -0.066 |
| y_1r2_close60m | 2.000 | 0.318 | 0.345 | -0.035 | 0.028 | 0.063 | 411.000 | -0.016 | 0.088 | 0.103 | -0.019 |
| y_1r2_240m | 2.000 | 0.323 | 0.350 | -0.053 | 0.047 | 0.100 | 413.000 | -0.072 | 0.094 | 0.166 | 0.019 |

## Yearly Pass Rate by Label (v3)

| target | pass_2025 | pass_2026 |
|---|---|---|
| y_1r4_180m | 0.347 | 0.000 |
| y_1r4_120m | 0.372 | 0.000 |
| y_1r4_60m | 0.222 | 0.000 |
| y_1r4_240m | 0.234 | 0.000 |
| y_1r4_close60m | 0.222 | 0.000 |
| y_1r2_120m | 0.251 | 0.000 |
| y_1r2_60m | 0.439 | 0.000 |
| y_1r2_180m | 0.289 | 0.049 |
| y_1r2_close60m | 0.038 | 0.000 |
| y_1r2_240m | 0.063 | 0.000 |

## Yearly Fail MLL Rate by Label (v3)

| target | fail_2025 | fail_2026 |
|---|---|---|
| y_1r4_180m | 0.050 | 0.000 |
| y_1r4_120m | 0.113 | 0.000 |
| y_1r4_60m | 0.000 | 0.000 |
| y_1r4_240m | 0.017 | 0.000 |
| y_1r4_close60m | 0.063 | 0.000 |
| y_1r2_120m | 0.046 | 0.295 |
| y_1r2_60m | 0.167 | 0.607 |
| y_1r2_180m | 0.159 | 0.492 |
| y_1r2_close60m | 0.054 | 0.115 |
| y_1r2_240m | 0.034 | 0.377 |

## Yearly Median PnL by Label (v3)

| target | pnl_2025 | pnl_2026 |
|---|---|---|
| y_1r4_180m | 1146.000 | -356.000 |
| y_1r4_120m | 1664.000 | -642.000 |
| y_1r4_60m | 467.000 | 430.000 |
| y_1r4_240m | 646.000 | 251.000 |
| y_1r4_close60m | 0.000 | -321.000 |
| y_1r2_120m | 1191.000 | -917.000 |
| y_1r2_60m | 2294.000 | -1470.000 |
| y_1r2_180m | 1485.000 | 170.000 |
| y_1r2_close60m | 606.000 | -317.000 |
| y_1r2_240m | 650.000 | -36.000 |

## Feature Importance (avg across all targets)

| feature | avg_gain_pct | type |
|---|---|---|
| price_vs_vwap_pct_abs | 25.3% | scale-invariant 🆕 |
| atr14_at_entry | 11.3% | original |
| adx_14_15m | 10.7% | original |
| breakout_strength_atr_ratio | 9.7% | scale-invariant 🆕 |
| price_vs_vwap_pct | 8.8% | original |
| orb_range_atr_ratio | 8.3% | original |
| orb_range_sq | 5.9% | scale-invariant 🆕 |
| breakout_strength_vs_orb | 4.6% | scale-invariant 🆕 |
| time_in_session_min | 3.8% | original |
| ema_slope_1h | 2.1% | original |
| breakout_strength | 1.9% | original |
| atr14_sq | 1.8% | scale-invariant 🆕 |
| day_of_week | 1.7% | original |
| session | 1.5% | original |
| breakout_side | 1.1% | original |
| breakout_strength_sq | 1.0% | scale-invariant 🆕 |
| adx_50_flag | 0.5% | scale-invariant 🆕 |
| orb_tf | 0.1% | original |

## Feature Distribution Drift: Training vs 2026

How well do scale-invariant features maintain consistent distributions?

| feature | train_median | 2026_median | ratio | stable |
|---|---|---|---|---|
| adx_14_15m | 25.6150 | 24.1562 | 0.94x | ✅ |
| adx_50_flag | 0.0000 | 0.0000 | inf | 🚨 |
| atr14_at_entry | 0.7076 | 3.5318 | 4.99x | ⚠️ |
| atr14_sq | 0.5034 | 12.4734 | 24.78x | 🚨 |
| breakout_side | 1.0000 | 1.0000 | 1.00x | ✅ |
| breakout_strength | 0.4000 | 1.6001 | 4.00x | ⚠️ |
| breakout_strength_atr_ratio | 0.7739 | 0.4490 | 0.58x | ✅ |
| breakout_strength_sq | 0.1600 | 2.5603 | 16.00x | 🚨 |
| breakout_strength_vs_orb | 0.1714 | 0.1002 | 0.58x | ✅ |
| day_of_week | 2.0000 | 2.0000 | 1.00x | ✅ |
| ema_slope_1h | 1.0000 | 1.0000 | 1.00x | ✅ |
| orb_range | 2.7000 | 16.3999 | 6.07x | 🚨 |
| orb_range_atr_ratio | 3.9002 | 4.4957 | 1.15x | ✅ |
| orb_range_sq | 7.2900 | 268.9568 | 36.89x | 🚨 |
| price_vs_vwap_pct | 0.0132 | 0.0780 | 5.92x | 🚨 |
| price_vs_vwap_pct_abs | 0.1001 | 0.1778 | 1.78x | ✅ |
| time_in_session_min | 37.0000 | 35.0000 | 0.95x | ✅ |

**Legend:**
- ✅ **Stable** = 2026 median within 0.5×-2.0× training median
- ⚠️ **Moderate drift** = 2026 median 2×-5× training median
- 🚨 **Severe drift** = 2026 median >5× training median

## Detailed Feature Distribution by Year

See `FEATURE_DISTRIBUTIONS.csv` for per-year medians/p5/p95 of all features.

## Notes

- Models trained on 2023-12-31 backward, early-stopped on 2024-01-01..2024-12-31
- Sample weighting: exponential decay half-life=2y
- Policy: dynamic rev/cont/skip with trend+ADX gates
- Holdout starts 2025-01-01 (2025-2026 — the high-volatility regime)
- **v2 baseline results** in this report are re-trained in same script (not imported from v2) — apples-to-apples comparison

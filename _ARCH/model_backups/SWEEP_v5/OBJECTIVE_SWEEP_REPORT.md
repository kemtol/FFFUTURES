# Objective Sweep v5 — Recent Regime Training Only

## Key Hypothesis

Gold regime changed dramatically from 2024 onward:
- Price tripled (~$1,800 → ~$3,000+)
- ATR14 exploded from $0.76 (training) to $3.53 (2026) — **4.6×**
- Training on 2010-2023 low-vol data HURTS generalization to 2026

**This sweep trains ONLY on 2024 data, calibrates on 2025-H1, and tests on 2025-H2 + 2026.**

## Key Changes from v4

| Aspect | v4 | v5 |
|--------|:--:|:--:|
| Train data | 2010 → 2023-12-31 (14 years) | **2024-01-01 → 2024-12-31 (1 year)** |
| Calibration | 2020 → 2023 (4 years) | **2025-H1 (6 months)** |
| Holdout | 2024+ | **2025-H2 + 2026** |
| Half-life decay | 2 years | **1 year** |
| Min data in leaf | 50 | **20** |
| Num boost rounds | 300 | **200** |

## Summary

| Metric | Value |
|--------|-------|
| Labels evaluated | 10 |
| Training data | 2024 (1 year of high-vol regime) |
| Calibration | 2025-H1 |
| Holdout | 2025-H2 + 2026 |
| Param grid | 128 combinations |
| Simulator | Refined (CT trade day + $3 commission) |

## Ranked Results (v5 — best params per label)

| target | rr | score | pass_rate | fail_mll_rate | median_end_pnl | avg_trades | windows |
|---|---|---|---|---|---|---|---|
| y_1r4_120m | 4.000 | 0.012 | 1.2% | 0.0% | 910.000 | 34.277 | 83 |
| y_1r2_240m | 2.000 | 0.012 | 1.2% | 0.0% | 558.000 | 13.446 | 83 |
| y_1r2_60m | 2.000 | 0.000 | 0.0% | 0.0% | 31.000 | 13.470 | 83 |
| y_1r2_120m | 2.000 | 0.000 | 0.0% | 0.0% | -336.000 | 8.530 | 83 |
| y_1r2_180m | 2.000 | 0.000 | 0.0% | 0.0% | -45.000 | 14.590 | 83 |
| y_1r2_close60m | 2.000 | 0.000 | 0.0% | 0.0% | 58.000 | 15.831 | 83 |
| y_1r4_60m | 4.000 | -0.084 | 1.2% | 9.6% | 307.000 | 25.651 | 83 |
| y_1r4_close60m | 4.000 | -0.169 | 0.0% | 16.9% | 146.000 | 22.434 | 83 |
| y_1r4_180m | 4.000 | -0.241 | 0.0% | 24.1% | 1099.000 | 43.446 | 83 |
| y_1r4_240m | 4.000 | -0.265 | 1.2% | 27.7% | 247.000 | 39.819 | 83 |

## Yearly Pass Rate by Label (v5)

| target | pass_2025 | pass_2026 | pass_rate |
|---|---|---|---|
| y_1r4_120m | 0.0% | 1.6% | 1.2% |
| y_1r2_240m | 0.0% | 1.6% | 1.2% |
| y_1r2_60m | 0.0% | 0.0% | 0.0% |
| y_1r2_120m | 0.0% | 0.0% | 0.0% |
| y_1r2_180m | 0.0% | 0.0% | 0.0% |
| y_1r2_close60m | 0.0% | 0.0% | 0.0% |
| y_1r4_60m | 0.0% | 1.6% | 1.2% |
| y_1r4_close60m | 0.0% | 0.0% | 0.0% |
| y_1r4_180m | 0.0% | 0.0% | 0.0% |
| y_1r4_240m | 0.0% | 1.6% | 1.2% |

## Yearly Fail MLL Rate by Label (v5)

| target | fail_mll_2025 | fail_mll_2026 | fail_mll_rate |
|---|---|---|---|
| y_1r4_120m | 0.0% | 0.0% | 0.0% |
| y_1r2_240m | 0.0% | 0.0% | 0.0% |
| y_1r2_60m | 0.0% | 0.0% | 0.0% |
| y_1r2_120m | 0.0% | 0.0% | 0.0% |
| y_1r2_180m | 0.0% | 0.0% | 0.0% |
| y_1r2_close60m | 0.0% | 0.0% | 0.0% |
| y_1r4_60m | 100.0% | 0.0% | 9.6% |
| y_1r4_close60m | 100.0% | 0.0% | 16.9% |
| y_1r4_180m | 0.0% | 26.2% | 24.1% |
| y_1r4_240m | 0.0% | 37.7% | 27.7% |

## Yearly Median PnL by Label (v5)

| target | pnl_2025 | pnl_2026 |
|---|---|---|
| y_1r4_120m | $-341 | $+1307 |
| y_1r2_240m | $+2196 | $-171 |
| y_1r2_60m | $+640 | $-318 |
| y_1r2_120m | $+1276 | $-439 |
| y_1r2_180m | $+640 | $-421 |
| y_1r2_close60m | $-45 | $-336 |
| y_1r4_60m | $-1678 | $+1013 |
| y_1r4_close60m | $-1090 | $+249 |
| y_1r4_180m | $+511 | $+2571 |
| y_1r4_240m | $-195 | $-208 |

## Best Params per Label

| target | target | rev_q | cont_q | rev_adx_min | cont_adx_max | risk_per_r_usd | daily_profit_cap |
|---|---|---|---|---|---|---|---|
| y_1r4_120m | y_1r4_120m | 0.600 | 0.750 | 40 | 30 | 100.000 | 1400.000 |
| y_1r2_240m | y_1r2_240m | 0.600 | 0.750 | 30 | 30 | 150.000 | 0.000 |
| y_1r2_60m | y_1r2_60m | 0.600 | 0.750 | 30 | 30 | 100.000 | 0.000 |
| y_1r2_120m | y_1r2_120m | 0.600 | 0.750 | 40 | 30 | 100.000 | 0.000 |
| y_1r2_180m | y_1r2_180m | 0.600 | 0.750 | 30 | 30 | 100.000 | 0.000 |
| y_1r2_close60m | y_1r2_close60m | 0.600 | 0.750 | 30 | 30 | 100.000 | 0.000 |
| y_1r4_60m | y_1r4_60m | 0.600 | 0.750 | 30 | 30 | 100.000 | 1400.000 |
| y_1r4_close60m | y_1r4_close60m | 0.600 | 0.750 | 30 | 30 | 100.000 | 0.000 |
| y_1r4_180m | y_1r4_180m | 0.750 | 0.750 | 30 | 30 | 100.000 | 0.000 |
| y_1r4_240m | y_1r4_240m | 0.600 | 0.750 | 40 | 30 | 100.000 | 1400.000 |

## Notes

- Training: only 2024 data (1 year, high-vol regime)
- Calibration: 2025-H1 (6 months, used for early stopping)
- Holdout: 2025-H2 + 2026 (entirely out-of-sample from training)
- Exponential decay: 1-year half-life (weights 2024-H2 > 2024-H1)
- Simulator: refined (CT trade day + $3 commission, same as v4)
- Features: identical to v3/v4 (18 total)

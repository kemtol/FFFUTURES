# Objective Sweep Report v2 — ORB (Extended Training)

Generated: 2026-04-26 22:19

## Key Changes from v1

| Parameter | v1 (original) | v2 (extended) |
|-----------|:-------------:|:-------------:|
| Training data | 2010-2021 | **2010-2023** |
| Validation/Calibration | 2022-2023 | **2024** |
| Holdout | 2024+ | **2025+** |
| ATR14 in training | $0.56-$0.76 | **$0.56-$0.73** |
| ADX in training | 27.2-29.6 | **27.2-44.4** |

## Summary

- Labels tested: 10
- Holdout: 2025-01-01 onward
- Topstep account: 50K (target $3,000, MLL $2,000)
- Scoring: `pass_rate - fail_mll_rate` (best params from 128-point grid)
- Best label: `y_1r4_180m` (score=+0.188)
- Labels with score > 0: 8

## Ranked Results (best params per label)

| target | rr | wr_rev_baseline | wr_cont_baseline | exp_rev_net | exp_cont_net | best_pass_rate | best_fail_mll | best_score | median_end_pnl | best_risk_per_r |
|---|---|---|---|---|---|---|---|---|---|---|
| y_1r4_180m | 4.000 | 0.177 | 0.206 | -0.187 | -0.042 | 0.229 | 0.041 | 0.188 | 432.000 | 100.000 |
| y_1r4_60m | 4.000 | 0.143 | 0.176 | -0.356 | -0.189 | 0.248 | 0.100 | 0.147 | 648.000 | 150.000 |
| y_1r4_240m | 4.000 | 0.180 | 0.208 | -0.170 | -0.032 | 0.185 | 0.044 | 0.141 | 541.000 | 100.000 |
| y_1r2_180m | 2.000 | 0.323 | 0.350 | -0.102 | -0.019 | 0.198 | 0.119 | 0.078 | 415.000 | 100.000 |
| y_1r4_120m | 4.000 | 0.171 | 0.198 | -0.217 | -0.078 | 0.179 | 0.103 | 0.075 | -245.000 | 100.000 |
| y_1r4_close60m | 4.000 | 0.151 | 0.180 | -0.313 | -0.169 | 0.163 | 0.094 | 0.069 | 274.500 | 150.000 |
| y_1r2_60m | 2.000 | 0.318 | 0.344 | -0.117 | -0.037 | 0.075 | 0.044 | 0.031 | 201.000 | 100.000 |
| y_1r2_120m | 2.000 | 0.322 | 0.350 | -0.104 | -0.021 | 0.129 | 0.116 | 0.013 | 413.000 | 100.000 |
| y_1r2_close60m | 2.000 | 0.318 | 0.345 | -0.116 | -0.035 | 0.088 | 0.103 | -0.016 | 654.000 | 100.000 |
| y_1r2_240m | 2.000 | 0.323 | 0.350 | -0.102 | -0.018 | 0.094 | 0.166 | -0.072 | 333.000 | 100.000 |

## Yearly Pass Rate by Label

| target | pass_2025 | pass_2026 |
|---|---|---|
| y_1r4_180m | 0.305 | 0.000 |
| y_1r4_60m | 0.331 | 0.000 |
| y_1r4_240m | 0.247 | 0.000 |
| y_1r2_180m | 0.264 | 0.000 |
| y_1r4_120m | 0.238 | 0.000 |
| y_1r4_close60m | 0.201 | 0.066 |
| y_1r2_60m | 0.100 | 0.000 |
| y_1r2_120m | 0.172 | 0.000 |
| y_1r2_close60m | 0.117 | 0.000 |
| y_1r2_240m | 0.126 | 0.000 |

## Yearly Fail MLL Rate by Label

| target | fail_2025 | fail_2026 |
|---|---|---|
| y_1r4_180m | 0.054 | 0.000 |
| y_1r4_60m | 0.042 | 0.049 |
| y_1r4_240m | 0.059 | 0.000 |
| y_1r2_180m | 0.046 | 0.443 |
| y_1r4_120m | 0.130 | 0.000 |
| y_1r4_close60m | 0.088 | 0.147 |
| y_1r2_60m | 0.000 | 0.131 |
| y_1r2_120m | 0.105 | 0.147 |
| y_1r2_close60m | 0.000 | 0.541 |
| y_1r2_240m | 0.121 | 0.393 |

## Yearly Median PnL by Label

| target | pnl_2025 | pnl_2026 |
|---|---|---|
| y_1r4_180m | 683.000 | -140.000 |
| y_1r4_60m | 1182.000 | 592.000 |
| y_1r4_240m | 1043.000 | -603.000 |
| y_1r2_180m | 1040.000 | -1343.000 |
| y_1r4_120m | -241.000 | -214.000 |
| y_1r4_close60m | 376.000 | 966.000 |
| y_1r2_60m | 671.000 | -426.000 |
| y_1r2_120m | 931.000 | -787.000 |
| y_1r2_close60m | 1233.000 | -1536.000 |
| y_1r2_240m | 824.000 | -806.000 |

## Notes

- Models trained on 2023-12-31 backward, early-stopped on 2024-01-01..2024-12-31
- Sample weighting: exponential decay half-life=2y
- Policy: dynamic rev/cont/skip with trend+ADX gates
- Holdout starts 2025-01-01 (2025-2026 — the high-volatility regime)

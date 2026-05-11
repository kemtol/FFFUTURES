# Objective Sweep Report v4 — Refined Topstep Simulator

## Key Changes from v3

1. **CT-based trading day boundaries** — `map_to_topstep_trade_day()`
   - Topstep trading day: 5:00 PM CT to 3:10 PM CT next day
   - Tokyo session (00:00–03:00 UTC) maps to PREVIOUS trading day
2. **Fixed $3.00 commission** (not 0.07R percentage)
3. **Shared module** — `pipeline/analysis/topstep_sim.py`
4. **Identical features** as v3 (11 baseline + 7 scale-invariant)

## Summary

| Metric | Value |
|--------|-------|
| Labels evaluated | 10 |
| Training data | ≤ 2023-12-31 |
| Calibration | 2020-01-01 → 2023-12-31 |
| Holdout | 2024-01-01+ |
| Param grid | 128 combinations |
| Simulator | Refined (CT trade day + $3 commission) |

## Ranked Results (v4 — best params per label)

| target | rr | score | pass_rate | fail_mll_rate | median_end_pnl | avg_trades | windows |
|---|---|---|---|---|---|---|---|
| y_1r2_180m | 2.000 | 0.064 | 22.7% | 16.3% | 1264.000 | 53.152 | 578 |
| y_1r4_240m | 4.000 | 0.042 | 18.2% | 14.0% | 608.500 | 38.656 | 578 |
| y_1r2_240m | 2.000 | 0.040 | 17.6% | 13.7% | 1136.500 | 45.066 | 578 |
| y_1r2_close60m | 2.000 | 0.033 | 11.9% | 8.7% | 429.000 | 39.007 | 578 |
| y_1r2_120m | 2.000 | 0.029 | 23.7% | 20.8% | 1051.500 | 53.974 | 578 |
| y_1r2_60m | 2.000 | 0.014 | 17.8% | 16.4% | 514.500 | 52.135 | 578 |
| y_1r4_120m | 4.000 | -0.003 | 23.2% | 23.5% | 307.000 | 33.619 | 578 |
| y_1r4_180m | 4.000 | -0.005 | 24.4% | 24.9% | 439.000 | 36.742 | 578 |
| y_1r4_close60m | 4.000 | -0.057 | 12.5% | 18.2% | 93.500 | 33.026 | 578 |
| y_1r4_60m | 4.000 | -0.189 | 17.8% | 36.7% | -39.500 | 35.239 | 578 |

## Yearly Pass Rate by Label (v4)

| target | pass_2024 | pass_2025 | pass_2026 | pass_rate |
|---|---|---|---|---|
| y_1r2_180m | 28.7% | 25.9% | 0.0% | 22.7% |
| y_1r4_240m | 26.2% | 17.2% | 1.6% | 18.2% |
| y_1r2_240m | 25.8% | 15.9% | 3.3% | 17.6% |
| y_1r2_close60m | 15.0% | 9.2% | 0.0% | 11.9% |
| y_1r2_120m | 28.7% | 28.5% | 0.0% | 23.7% |
| y_1r2_60m | 25.4% | 17.6% | 0.0% | 17.8% |
| y_1r4_120m | 35.0% | 19.2% | 0.0% | 23.2% |
| y_1r4_180m | 32.5% | 21.8% | 0.0% | 24.4% |
| y_1r4_close60m | 22.5% | 7.1% | 1.6% | 12.5% |
| y_1r4_60m | 23.3% | 17.6% | 0.0% | 17.8% |

## Yearly Fail MLL Rate by Label (v4)

| target | fail_mll_2024 | fail_mll_2025 | fail_mll_2026 | fail_mll_rate |
|---|---|---|---|---|
| y_1r2_180m | 13.8% | 13.0% | 49.2% | 16.3% |
| y_1r4_240m | 5.4% | 17.6% | 26.2% | 14.0% |
| y_1r2_240m | 2.9% | 20.1% | 39.3% | 13.7% |
| y_1r2_close60m | 5.4% | 15.1% | 1.6% | 8.7% |
| y_1r2_120m | 12.5% | 22.2% | 60.7% | 20.8% |
| y_1r2_60m | 8.3% | 22.6% | 34.4% | 16.4% |
| y_1r4_120m | 14.2% | 33.1% | 21.3% | 23.5% |
| y_1r4_180m | 22.5% | 29.7% | 31.1% | 24.9% |
| y_1r4_close60m | 15.8% | 25.1% | 8.2% | 18.2% |
| y_1r4_60m | 27.1% | 41.8% | 32.8% | 36.7% |

## Yearly Median PnL by Label (v4)

| target | pnl_2024 | pnl_2025 | pnl_2026 |
|---|---|---|---|
| y_1r2_180m | $+1638 | $+1495 | $-305 |
| y_1r4_240m | $+1513 | $+513 | $-736 |
| y_1r2_240m | $+1660 | $+774 | $+595 |
| y_1r2_close60m | $+971 | $+116 | $-251 |
| y_1r2_120m | $+1508 | $+1105 | $-247 |
| y_1r2_60m | $+1262 | $+268 | $-538 |
| y_1r4_120m | $+1402 | $-32 | $-384 |
| y_1r4_180m | $+1872 | $+161 | $-442 |
| y_1r4_close60m | $+777 | $-90 | $-221 |
| y_1r4_60m | $+1298 | $-399 | $-633 |

## Best Params per Label

| target | target | rev_q | cont_q | rev_adx_min | cont_adx_max | risk_per_r_usd | daily_profit_cap |
|---|---|---|---|---|---|---|---|
| y_1r2_180m | y_1r2_180m | 0.750 | 0.600 | 30 | 100 | 100.000 | 1400.000 |
| y_1r4_240m | y_1r4_240m | 0.750 | 0.750 | 30 | 30 | 100.000 | 1400.000 |
| y_1r2_240m | y_1r2_240m | 0.750 | 0.600 | 30 | 30 | 100.000 | 1400.000 |
| y_1r2_close60m | y_1r2_close60m | 0.750 | 0.600 | 30 | 30 | 100.000 | 1400.000 |
| y_1r2_120m | y_1r2_120m | 0.750 | 0.600 | 30 | 100 | 100.000 | 1400.000 |
| y_1r2_60m | y_1r2_60m | 0.600 | 0.600 | 40 | 100 | 100.000 | 1400.000 |
| y_1r4_120m | y_1r4_120m | 0.750 | 0.750 | 30 | 30 | 100.000 | 1400.000 |
| y_1r4_180m | y_1r4_180m | 0.600 | 0.750 | 40 | 30 | 100.000 | 1400.000 |
| y_1r4_close60m | y_1r4_close60m | 0.750 | 0.750 | 30 | 30 | 100.000 | 1400.000 |
| y_1r4_60m | y_1r4_60m | 0.750 | 0.750 | 30 | 30 | 100.000 | 1400.000 |

## Notes

- Simulator uses CT-based trading day boundaries (5PM CT → 3:10PM CT next day)
- Commission: fixed $3.00/round-turn (entry + exit + slippage)
- MLL trailing: tracks highest end-of-day PnL, locked at starting balance
- Consistency rule: best day < 50% of total profit at pass time
- Training: exponential time decay, half-life = 2 years
- Walk-forward: rolling 20-day windows, strict temporal holdout

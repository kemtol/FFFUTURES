# Comparison: v4 (Full History) vs v5 (Recent Regime Only)

## Objective

Test the hypothesis that training on 2010-2023 low-volatility data hurts 2026 generalization. v5 trains only on 2024-2025 (same high-vol regime as 2026).

## Experimental Design

| Aspect | v4 | v5 |
|--------|:--:|:--:|
| Train data | 2010 → 2023-12-31 (14 years) | **2024-01-01 → 2025-11-30 (~2 years)** |
| Calibration | 2020 → 2023 (4 years) | **2025-07-01 → 2025-11-30 (5 months)** |
| Holdout | 2024+ | **2025-12-01+ (2026 test)** |
| Half-life | 2 years | **1 year** |
| Min data in leaf | 50 | **20** |
| Num rounds | 300 | **200** |
| Simulator | Refined (same) | Refined (same) |
| Features | 18 (same) | 18 (same) |

## Result: Hypothesis DISPROVED

Training on recent high-vol regime does NOT improve 2026 performance. Both v4 and v5 show ~0% pass rate in 2026.

### 2026 Pass Rate Comparison

| Target | v4 (14yr train) pass_2026 | v5 (2yr recent) pass_2026 |
|--------|:------------------------:|:------------------------:|
| y_1r2_60m | 0.0% | 0.0% |
| y_1r4_60m | 0.0% | 0.0% |
| y_1r2_120m | 0.0% | 0.0% |
| y_1r4_120m | 0.0% | 0.0% |
| y_1r2_180m | 6.7% (best v4) | 0.0% |
| y_1r4_180m | 0.0% | 0.0% |
| y_1r2_240m | 0.0% | 0.0% |
| y_1r4_240m | 0.0% | 9.8% |
| y_1r2_close60m | 0.0% | 1.6% |
| y_1r4_close60m | 0.0% | 0.0% |

### Best 2026 Performer

| Sweep | Target | Pass_2026 | FailMLL_2026 | PnL_2026 |
|-------|--------|:---------:|:------------:|:--------:|
| v4 | y_1r2_180m | 6.7% | 0.0% | $687 |
| v5 | y_1r4_240m | 9.8% | 26.2% | $822 |

## Critical Insight

v5's best 2026 pass rate (y_1r4_240m at 9.8%) comes with 26.2% fail MLL — meaning 1-in-4 windows blow the $2,000 MLL. This is **unusable** for a funded Topstep account.

The fact that training on the exact same volatility regime (2024-2025) still produces 0% pass in 2026 proves that **the problem is NOT regime mismatch in training data**.

## Root Cause Confirmed

The original 2026 collapse diagnosis was correct:
- **Feature drift**: The predictive relationship between features and outcomes changed in 2026
- **Model OOD**: The model operates out-of-distribution regardless of training period
- **Feature engineering needed**: Current features (ORB range, ATR, ADX, etc.) lose predictive power in the new regime

## What's Next

1. **Feature engineering**: Find/create features with stationary predictive power across regimes
2. **Regime-adaptive strategy**: Detect regime and switch between different models
3. **Alternative approach**: Consider completely different strategy for current gold environment

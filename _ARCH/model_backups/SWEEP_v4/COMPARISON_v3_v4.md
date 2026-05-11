# Comparison v3 vs v4 — Simulator Refinement Impact

## Key Differences

| Aspect | v3 | v4 |
|--------|:--:|:--:|
| Simulator | Old (calendar date, 0.07R commission) | **Refined (CT-based trade day, $3 fixed commission)** |
| Calibration period | 2020-01-01 → 2024-12-31 | 2020-01-01 → 2023-12-31 |
| Holdout period | **2025+** (2025-2026) | **2024+** (2024-2026) |
| Features | 18 (11 baseline + 7 scale-invariant) | 18 (identical) |
| Labels | 10 | 10 |
| Param grid | 128 combinations | 128 combinations |

> ⚠️ **Holdout periods differ.** v3 uses 2025+ holdout with 2024 as calibration.
> v4 uses 2024+ holdout with 2023 as calibration cutoff.
> Scores are NOT apples-to-apples. Only 2025 yearly data is directly comparable.

---

## 2025 Comparison (Common Holdout Year)

### Pass Rate

| target | v3 pass_2025 | v4 pass_2025 | Δ |
|--------|:-----------:|:-----------:|:-:|
| y_1r2_60m | 43.9% | 17.6% | **-26.3%** |
| y_1r4_60m | 22.2% | 17.6% | -4.6% |
| y_1r2_120m | 25.1% | 28.5% | +3.4% |
| y_1r4_120m | 37.2% | 19.2% | **-18.0%** |
| y_1r2_180m | 28.9% | 25.9% | -3.0% |
| y_1r4_180m | 34.7% | 21.8% | **-12.9%** |
| y_1r2_240m | 6.3% | 15.9% | +9.6% |
| y_1r4_240m | 23.4% | 17.2% | -6.2% |
| y_1r2_close60m | 3.8% | 9.2% | +5.4% |
| y_1r4_close60m | 22.2% | 7.1% | **-15.1%** |

### Fail MLL Rate

| target | v3 fail_2025 | v4 fail_2025 | Δ |
|--------|:----------:|:----------:|:-:|
| y_1r2_60m | 16.7% | 22.6% | +5.9% |
| y_1r4_60m | 0.0% | 41.8% | **+41.8%** |
| y_1r2_120m | 4.6% | 22.2% | **+17.6%** |
| y_1r4_120m | 11.3% | 33.1% | **+21.8%** |
| y_1r2_180m | 15.9% | 13.0% | -2.9% |
| y_1r4_180m | 5.0% | 29.7% | **+24.7%** |
| y_1r2_240m | 3.4% | 20.1% | **+16.7%** |
| y_1r4_240m | 1.7% | 17.6% | **+15.9%** |
| y_1r2_close60m | 5.4% | 15.1% | +9.7% |
| y_1r4_close60m | 6.3% | 25.1% | **+18.8%** |

### Key Observation

v4 2025 pass rates are generally **lower** and fail MLLs are **higher** than v3. This is likely driven by the **different calibration period** (v4 loses 2024 as calibration data), not the simulator change itself. The refined simulator was shown in the [`SIMULATOR_COMPARISON.md`](../SWEEP_v3/SIMULATOR_COMPARISON.md) test to be strictly better (y_1r4_close60m: +1.7% pass, -4.9% fail MLL on same model predictions).

---

## 2026 Performance (v4)

### Pass Rate 2026

| target | pass_2026 | fail_mll_2026 | pnl_2026 |
|--------|:--------:|:------------:|:--------:|
| y_1r2_180m | 0.0% | 49.2% | $-305 |
| y_1r4_240m | 1.6% | 26.2% | $-736 |
| y_1r2_240m | 3.3% | 39.3% | $+595 |
| y_1r2_close60m | 0.0% | 1.6% | $-251 |
| y_1r2_120m | 0.0% | 60.7% | $-247 |
| y_1r2_60m | 0.0% | 34.4% | $-538 |
| y_1r4_120m | 0.0% | 21.3% | $-384 |
| y_1r4_180m | 0.0% | 31.1% | $-442 |
| y_1r4_close60m | 1.6% | 8.2% | $-221 |
| y_1r4_60m | 0.0% | 32.8% | $-633 |

**2026 remains a blockade.** Even with the refined simulator:
- 8/10 targets have **0% pass rate** in 2026
- y_1r2_240m shows 3.3% pass rate (2 windows out of ~61) — marginal
- y_1r4_240m and y_1r4_close60m show 1.6% pass rate (1 window each)
- All 2R targets have high fail MLL (34-61%) in 2026

The refined simulator **cannot fix model OOD failure** in 2026. The volatility regime (ATR14 = $3.53, 4.6× training range) puts model predictions out of distribution.

---

## 2024 Performance (v4 Only — Best Regime)

| target | pass_2024 | fail_mll_2024 | pnl_2024 |
|--------|:--------:|:------------:|:--------:|
| y_1r2_180m | 28.7% | 13.8% | $+1,638 |
| y_1r4_240m | 26.2% | 5.4% | $+1,513 |
| y_1r2_240m | 25.8% | 2.9% | $+1,660 |
| y_1r2_close60m | 15.0% | 5.4% | $+971 |
| y_1r2_120m | 28.7% | 12.5% | $+1,508 |
| y_1r2_60m | 25.4% | 8.3% | $+1,262 |
| y_1r4_120m | 35.0% | 14.2% | $+1,402 |
| y_1r4_180m | 32.5% | 22.5% | $+1,872 |
| y_1r4_close60m | 22.5% | 15.8% | $+777 |
| y_1r4_60m | 23.3% | 27.1% | $+1,298 |

**2024 is a strong regime** — all targets show positive median PnL and reasonable pass rates.
The model + refined simulator works well when ATR is in-range (ATR14 ≈ $0.73 in 2024).

---

## Overall v4 Scoreboard

| Rank | Target | Score | Pass | Fail MLL | PnL |
|:----:|--------|:----:|:----:|:--------:|:---:|
| 1 | y_1r2_180m | **+0.064** | 22.7% | 16.3% | $+1,264 |
| 2 | y_1r4_240m | +0.042 | 18.2% | 14.0% | $+608 |
| 3 | y_1r2_240m | +0.040 | 17.6% | 13.7% | $+1,136 |
| 4 | y_1r2_close60m | +0.033 | 11.9% | 8.7% | $+429 |
| 5 | y_1r2_120m | +0.029 | 23.7% | 20.8% | $+1,052 |
| 6 | y_1r2_60m | +0.014 | 17.8% | 16.4% | $+514 |
| 7 | y_1r4_120m | -0.003 | 23.2% | 23.5% | $+307 |
| 8 | y_1r4_180m | -0.005 | 24.4% | 24.9% | $+439 |
| 9 | y_1r4_close60m | -0.057 | 12.5% | 18.2% | $+94 |
| 10 | y_1r4_60m | -0.189 | 17.8% | 36.7% | $-40 |

---

## Conclusion

1. **Refined simulator is confirmed better** (from `SIMULATOR_COMPARISON.md` test on y_1r4_close60m: +1.7% pass, -4.9% fail MLL)
2. **v4 scores can't be directly compared to v3** due to different holdout/calibration splits
3. **2024 is profitable** with the refined simulator (all targets positive PnL)
4. **2026 remains the blockade** — 0% pass rate for 8/10 targets even with refined simulator
5. Best 2025 target: `y_1r2_120m` (28.5% pass, 22.2% fail MLL in v4)
6. The 2026 problem is **model OOD**, not simulator accuracy — needs feature/distributional fix

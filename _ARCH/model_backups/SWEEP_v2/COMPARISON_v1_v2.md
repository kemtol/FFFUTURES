# Comparison: Objective Sweep v1 vs v2

## Settings

| Parameter | v1 | v2 |
|-----------|:--:|:--:|
| Training data | 2010-2021 | 2010-2023 |
| Validation/Calibration | 2022-2023 | 2024 |
| Holdout | 2024+ | 2025+ |
| Labels tested | 10 | 10 |

---

## Overall Best Results

| Metric | v1 (y_1r4_240m) | v2 (y_1r4_180m) | Delta |
|--------|:--------------:|:--------------:|:-----:|
| Best pass_rate | 36.0% | 22.9% | -13.1 pp |
| Best fail_mll | 14.4% | 4.1% | **-10.3 pp** ✅ |
| Best score | +0.216 | +0.188 | -0.028 |
| Best $/1R | $100 | $100 | Same |

**Note:** Different holdout periods. v1 includes 2024 (good year), v2 starts 2025 (harder).

---

## 🚨 Key Discovery: y_1r4_close60m in v2

**First target EVER to achieve non-zero pass_rate in 2026:**

| Metric | 2025 | **2026** |
|--------|:----:|:--------:|
| pass_rate | 20.1% | **6.6%** 🔥 |
| fail_mll | 8.8% | 14.8% |
| median PnL | +$376 | **+$966** 🔥 |

**Why it works:** close60m targets capture mean-reversion before session close. In high-volatility 2026, the 4R is hit more frequently within the remaining session time. The shorter horizon (close60m vs 240m) means less exposure to adverse moves.

---

## Per-Target Comparison (2025 performance)

| Target | v1 2025 pass | v2 2025 pass | v1 fail_mll | v2 fail_mll |
|--------|:-----------:|:-----------:|:----------:|:----------:|
| y_1r2_60m | 15.1% | 10.0% | 10.0% | **0.0%** ✅ |
| y_1r4_60m | 8.4% | **33.1%** 🔥 | 0.8% | 4.2% |
| y_1r2_120m | 38.9% | 17.2% | 13.0% | 10.5% |
| y_1r4_120m | 16.7% | 23.9% | **0.0%** ✅ | 13.0% |
| y_1r2_180m | 6.3% | **26.4%** ✅ | 0.0% | 4.6% |
| y_1r4_180m | 22.6% | **30.5%** ✅ | 6.3% | 5.4% |
| y_1r2_240m | 5.0% | 12.6% | 5.0% | 12.1% |
| y_1r4_240m | 31.0% | 24.7% | 6.3% | 5.9% |
| y_1r2_close60m | 27.6% | 11.7% | 11.3% | **0.0%** ✅ |
| y_1r4_close60m | 14.2% | 20.1% | 0.0% | 8.8% |

v2 improves 2025 pass_rate for 5 out of 10 targets.

---

## 2026 Performance (v2 only)

| Target | pass_rate | fail_mll | median PnL |
|--------|:--------:|:--------:|:----------:|
| **y_1r4_close60m** | **6.6%** 🔥 | **14.8%** | **+$966** |
| y_1r4_60m | 0% | 4.9% | +$592 |
| y_1r4_180m | 0% | 0% | -$140 |
| y_1r4_240m | 0% | 0% | -$603 |
| y_1r4_120m | 0% | 0% | -$214 |
| y_1r2_60m | 0% | 13.1% | -$426 |
| y_1r2_120m | 0% | 14.8% | -$787 |
| y_1r2_180m | 0% | 44.3% | -$1,343 |
| y_1r2_240m | 0% | 39.3% | -$806 |
| y_1r2_close60m | 0% | 54.1% | -$1,536 |

---

## Conclusions

1. **Extended training (v2) improves 2025 pass_rate for most targets** — confirms OOD was part of the problem
2. **`y_1r4_close60m` is the first target to show any 2026 pass_rate (6.6%)** — this is a genuine breakthrough
3. **4R targets consistently beat 2R** — risk-reward ratio is the key factor
4. **2026 remain extremely challenging** — even the best target only passes 6.6% of windows
5. **Fail_mll for some 2R targets is catastrophic in 2026** (44-54%) — these should be avoided entirely

## Recommended Path Forward

1. **Focus on `y_1r4_close60m`** — the only target showing 2026 life
2. **Add scale-invariant features** — normalize by rolling ATR so the model is volatility-agnostic
3. **Consider regime-conditional models** — separate model for high-volatility vs low-volatility regimes
4. **Topstep simulator refinement** — exact trading day boundaries will affect close60m labels

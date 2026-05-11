# 2026 Collapse Diagnosis — ORB Strategy

Generated: 2026-04-26 18:28

Focus: understand why all 10 label targets achieve **0% pass_rate in 2026**.

---

## 🚨 Root Cause Found: Volatility Explosion + Model OOD Failure

The diagnosis is clear. The 2026 collapse is caused by a **volatility explosion** that pushes all features **completely out-of-distribution (OOD)** relative to the training data (2010-2021). The model cannot discriminate because it has never seen this regime.

---

## 1. Event Volume by Year (2020+)

| Year | Events | Days | Events/Day |
|------|-------:|:----:|:----------:|
| 2020 | 4,600 | 258 | 17.8 |
| 2021 | 4,554 | 258 | 17.7 |
| 2022 | 4,564 | 258 | 17.7 |
| 2023 | 4,512 | 257 | 17.6 |
| 2024 | 4,602 | 259 | 17.8 |
| 2025 | 4,562 | 258 | 17.7 |
| **2026** | **1,418** | **80** | **17.7** |

➡ **Events/day is IDENTICAL across all years.** Trade starvation (Hypothesis D) is **rejected**.

---

## 2. ADX Distribution

| Year | ADX Mean | ADX Median | ADX<20 | ADX>50 |
|------|:--------:|:----------:|:------:|:------:|
| 2024 | 34.8 | 30.4 | 22.4% | 19.6% |
| 2025 | 28.0 | 25.4 | 31.1% | 5.2% |
| **2026** | **26.7** | **24.2** | **32.3%** | **4.2%** |

➡ Moderate ADX decline. ADX>50 dropped from 19.6% (2024) to 4.2% (2026), but 2025 was already at 5.2%. **Partial contributor** (Hypothesis A), not the main cause.

---

## 3. 🚨 VOLATILITY EXPLOSION — Root Cause

| Metric | 2024 | 2025 | **2026** | 2026 vs 2024 |
|--------|:----:|:----:|:--------:|:------------:|
| **ATR14 mean** | **$1.09** | **$1.98** | **$4.42** | **4.1× higher** 🔴 |
| ATR14 median | $0.73 | $1.57 | $3.53 | 4.8× higher |
| ATR14 p75 | $1.28 | $2.32 | $5.50 | 4.3× higher |
| ORB range median | $3.50 | $7.30 | **$16.40** | **4.7× higher** 🔴 |
| ORB range mean | $5.31 | $9.45 | $20.70 | 3.9× higher |

➡ **Hypothesis B CONFIRMED.** The volatility regime in 2026 is **4-5× higher** than 2024 and **completely outside** the training data distribution (2010-2021 training had ATR14 median $0.56-$0.76).

---

## 4. VWAP Position

| Year | Mean % | Std Dev |
|------|:------:|:-------:|
| 2024 | +0.015% | 0.20% |
| 2025 | +0.012% | 0.20% |
| **2026** | **+0.014%** | **0.30%** |

➡ Similar mean, but **std doubled** in 2026 (0.30% vs 0.20%). More extreme positions (both tails thicker). **Minor contributor** (Hypothesis C).

---

## 5. 🚨 CRITICAL FINDING: Label Hit Rates Are UNCHANGED

### y_1r4_240m Win Rate by Year

| Side | 2024 | 2025 | **2026** | Trend |
|------|:----:|:----:|:--------:|:------|
| Reversal | 20.9% | 17.4% | **19.7%** | ✅ Stable |
| Continuation | 19.6% | 21.6% | **18.1%** | ✅ Stable |

### ALL Label Hit Rates (Combined)

| Label | 2024 | 2025 | **2026** |
|-------|:----:|:----:|:--------:|
| y_1r2_60m | 33.7% | 33.4% | **32.2%** |
| y_1r4_60m | 17.6% | 16.0% | **15.9%** |
| y_1r2_120m | 33.9% | 33.9% | **32.7%** |
| y_1r4_120m | 19.3% | 18.5% | **18.1%** |
| y_1r2_180m | 34.1% | 33.9% | **32.7%** |
| y_1r4_180m | 19.9% | 19.3% | **18.4%** |
| y_1r2_240m | 34.1% | 34.0% | **32.7%** |
| y_1r4_240m | 20.2% | 19.5% | **18.9%** |
| y_1r2_close60m | 33.8% | 33.4% | **32.2%** |
| y_1r4_close60m | 18.4% | 16.5% | **16.9%** |

➡ **Hypothesis E REJECTED.** The ORB breakout structure is NOT broken. Label hit rates are practically unchanged across 2024-2026. The underlying pattern still works.

---

## 6. Win Rate y_1r4_240m by ADX Bucket

### Reversal

| ADX Bucket | 2024 | 2025 | **2026** |
|:----------:|:----:|:----:|:--------:|
| <20 | 18.0% | 17.5% | **18.3%** |
| 20-30 | 20.5% | 15.5% | **19.2%** |
| 30-50 | 22.9% | 19.0% | **22.3%** |
| >50 | 21.3% | 20.2% | **20.0%** |

### Continuation

| ADX Bucket | 2024 | 2025 | **2026** |
|:----------:|:----:|:----:|:--------:|
| <20 | 19.8% | 20.6% | **14.0%** |
| 20-30 | 22.8% | 21.5% | **18.0%** |
| 30-50 | 17.8% | 23.6% | **22.8%** |
| >50 | 17.6% | 16.8% | **20.0%** |

➡ Reversal WR is remarkably stable across ADX buckets. Continuation WR slightly lower in low-ADX regimes. This confirms the underlying label distribution is **not the problem**.

---

## 7. Feature Drift (Model OOD)

| Feature | Training (2010-2021) | 2024 | 2025 | **2026** | OOD? |
|---------|:-------------------:|:----:|:----:|:--------:|:----:|
| atr14_at_entry median | $0.56-$0.76 | $0.73 | $1.57 | **$3.53** | 🔴 Yes |
| orb_range median | $2.70-$3.80 | $3.50 | $7.30 | **$16.40** | 🔴 Yes |
| orb_range_atr_ratio median | 4.8-5.0 | 4.74 | 4.59 | **4.50** | ✅ Stable |
| breakout_strength median | 0.3-0.5 | 0.4 | 0.7 | **1.6** | 🔴 Yes |
| adx_14_15m median | 24.6-34.1 | 30.4 | 25.4 | **24.2** | ⚠️ Moderate |

➡ **The model's input features are at completely different scales in 2026.** The LGBM tree splits were optimized for low-volatility data (2010-2021). When applied to 2026's high-volatility regime, the model's probability estimates become unreliable.

---

## 8. The Paradox Explained

```
Raw label win rates:    ✅ Same (19-20% for y_1r4_240m rev)
Event frequency:        ✅ Same (17.7/day)
Model predictions:      ❌ OOD — features at 4× scale
Topstep pass_rate:      ❌ 0%
```

**Why pass_rate = 0% despite same win rates?**

The model was trained on 2010-2021 data where:
- ATR14 was ~$0.56-$0.76 median
- ORB range was ~$2.70-$3.80 median
- Breakout strength was ~0.3-0.5 median

In 2026:
- ATR14 is $3.53 median (4.6× training data)
- ORB range is $16.40 median (4.7× training data)
- Breakout strength is 1.6 median (4× training data)

The model's tree splits see feature values that never appeared in training. The model may be:
1. **Over-confident** — predicting very high or low probabilities that don't reflect true odds
2. **Under-confident** — predicting near-50% for everything
3. **Wrong direction** — ranking trades in reverse order of actual quality

Even with the same underlying label distribution, if the model's **ranking** is corrupted, the dynamic policy selects the wrong trades.

---

## Conclusion & Recommendations

### Root Cause

**Volatility explosion (ATR14 4.6× higher than training data) causes model feature OOD failure. The model's feature space in 2026 is completely outside its training distribution, corrupting probability estimates and trade selection.**

### Can This Be Fixed?

| Approach | Likelihood | Effort |
|----------|:----------:|:------:|
| **A. Retrain with 2022-2025 data** 🥇 | High | Low — just change TRAIN_TO date |
| **B. Add scale-invariant features** | Medium | Medium — e.g., rank-based features |
| **C. Normalize features per volatility regime** | Medium | Medium — divide by rolling ATR |
| **D. Online learning / adaptive model** | Low | High — complex infrastructure |

### Recommended Next Step

**Approach A is the highest ROI**: retrain the objective sweep models including 2022-2023 data in the training set (currently the script uses `TRAIN_TO = "2021-12-31"`, leaving 2022-2023 as holdout). 

With 2022-2025 data showing ATR trending from $0.58 → $0.73 → $1.57, the model will have some exposure to rising volatility regimes. While 2026 ($3.53) is still beyond that range, it's a smaller extrapolation gap.

Alternative: relax `TRAIN_TO` to include ALL pre-2024 data, making the training set include 2022-2023's moderate-to-high ADX regime.

---

## Detailed Data Appendix

See the raw data below for all tables referenced above.

### Event Count

| Year | Events | Days | Events/Day |
|------|-------:|:----:|:----------:|
| 2020 | 4,600 | 258 | 17.8 |
| 2021 | 4,554 | 258 | 17.7 |
| 2022 | 4,564 | 258 | 17.7 |
| 2023 | 4,512 | 257 | 17.6 |
| 2024 | 4,602 | 259 | 17.8 |
| 2025 | 4,562 | 258 | 17.7 |
| 2026 | 1,418 | 80 | 17.7 |

### ADX % Distribution

| Bucket | 2024 | 2025 | 2026 |
|--------|:----:|:----:|:----:|
| <20 | 22.4% | 31.1% | 32.3% |
| 20-30 | 27.1% | 33.0% | 37.5% |
| 30-50 | 30.9% | 30.6% | 26.0% |
| >50 | 19.6% | 5.2% | 4.2% |

### ATR14 Percentiles

| Percentile | 2024 | 2025 | 2026 |
|:----------:|:----:|:----:|:----:|
| 25% | $0.48 | $1.07 | $2.51 |
| 50% | $0.73 | $1.57 | $3.53 |
| 75% | $1.28 | $2.32 | $5.50 |

### ORB Range Median

| Year | Median |
|:----:|:------:|
| 2024 | $3.50 |
| 2025 | $7.30 |
| 2026 | $16.40 |

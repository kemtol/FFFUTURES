# Comparison: v2 (Baseline) vs v3 (Scale-Invariant Features)

Generated: 2026-04-26

## TL;DR

**Scale-invariant features improve 8/10 targets** (v3_score higher than v2_score), but **2026 pass_rate remains 0% for all targets** except `y_1r2_180m` (4.9%).

The most impactful new feature is `price_vs_vwap_pct_abs` — **#1 feature at 25.3% average gain** — which maintains a stable distribution across regimes (1.78× ratio). However, even with better features, the model cannot generalize to 2026's extreme volatility regime.

## AB Test Results

| target | v3_score | v2_score | delta | verdict |
|--------|:-------:|:-------:|:-----:|:-------:|
| y_1r4_180m | **+0.223** | +0.188 | +0.035 | ✅ improved |
| y_1r4_120m | **+0.198** | +0.075 | +0.122 | ✅ **best improvement** |
| y_1r4_60m | **+0.166** | +0.147 | +0.019 | ✅ |
| y_1r4_240m | **+0.163** | +0.141 | +0.022 | ✅ |
| y_1r4_close60m | **+0.119** | +0.069 | +0.050 | ✅ |
| y_1r2_120m | **+0.097** | +0.013 | +0.085 | ✅ |
| y_1r2_60m | **+0.056** | +0.031 | +0.025 | ✅ |
| y_1r2_180m | +0.013 | **+0.078** | -0.066 | ❌ degraded |
| y_1r2_close60m | -0.035 | **-0.016** | -0.019 | ❌ |
| y_1r2_240m | **-0.053** | -0.072 | +0.019 | ✅ (less negative) |

**Overall:** 8/10 improved, 2/10 degraded

## 2026 Pass Rate: The Hard Problem

| target | v3_2025_pass | v3_2026_pass | v2_2026_pass | verdict |
|--------|:-----------:|:-----------:|:-----------:|:-------:|
| y_1r2_180m | 28.9% | **4.9%** | 0.0% | ✅ **only target with any 2026 pass** |
| y_1r4_60m | 22.2% | 0.0% | 0.0% | ➡️ same (v2 also 0%) |
| y_1r4_120m | 37.2% | 0.0% | 0.0% | ➡️ same |
| y_1r4_180m | 34.7% | 0.0% | 0.0% | ➡️ same |
| y_1r4_240m | 23.4% | 0.0% | 0.0% | ➡️ same |
| y_1r4_close60m | 22.2% | 0.0% | **6.6%** | ❌ lost v2's 2026 edge |
| y_1r2_60m | 43.9% | 0.0% | 0.0% | ➡️ same |
| y_1r2_120m | 25.1% | 0.0% | 0.0% | ➡️ same |
| y_1r2_close60m | 3.8% | 0.0% | 0.0% | ➡️ same |
| y_1r2_240m | 6.3% | 0.0% | 0.0% | ➡️ same |

**Key finding:** `y_1r2_180m` shows the first-ever 2026 pass (4.9%), but at 49.2% fail_mll — unacceptably high.

## Feature Importance (New Features Only)

| Rank | New Feature | Avg Gain % | Description |
|:----:|-------------|:----------:|-------------|
| 1 | `price_vs_vwap_pct_abs` 🆕 | **25.3%** | Absolute VWAP deviation — #1 feature overall |
| 2 | `breakout_strength_atr_ratio` 🆕 | **9.7%** | Breakout strength normalized by ATR — #4 overall |
| 3 | `orb_range_sq` 🆕 | 5.9% | Squared ORB range for non-linear effects |
| 4 | `breakout_strength_vs_orb` 🆕 | 4.6% | Fraction of ORB range covered by breakout |
| 5 | `atr14_sq` 🆕 | 1.8% | Squared ATR — minor impact |
| 6 | `breakout_strength_sq` 🆕 | 1.0% | Squared breakout strength — minor |
| 7 | `adx_50_flag` 🆕 | 0.5% | Binary ADX>50 flag — very low importance |

## Feature Distribution Stability

| Feature | Type | Train Median | 2026 Median | Ratio | Stable? |
|---------|:----:|:-----------:|:----------:|:----:|:-------:|
| `price_vs_vwap_pct_abs` | 🆕 scale-inv | 0.100 | 0.178 | **1.78×** | ✅ |
| `breakout_strength_atr_ratio` | 🆕 scale-inv | 0.774 | 0.449 | **0.58×** | ✅ |
| `breakout_strength_vs_orb` | 🆕 scale-inv | 0.171 | 0.100 | **0.58×** | ✅ |
| `orb_range_atr_ratio` | original | 3.900 | 4.496 | **1.15×** | ✅ |
| `adx_14_15m` | original | 25.6 | 24.2 | **0.94×** | ✅ |
| `atr14_at_entry` | original | 0.708 | 3.532 | **4.99×** | ⚠️ |
| `breakout_strength` | original | 0.400 | 1.600 | **4.00×** | ⚠️ |
| `orb_range` | original | 2.700 | 16.400 | **6.07×** | 🚨 |
| `price_vs_vwap_pct` | original | 0.013 | 0.078 | **5.92×** | 🚨 |

**Scale-invariant features maintain stable distributions** (0.58×-1.78×). Original raw features still show 4-6× drift.

## Why Scale-Invariant Features Didn't Fix 2026

Despite the improved feature set:

1. **`price_vs_vwap_pct_abs`** became the #1 feature at 25% gain, replacing raw features
2. **Overall scores improved** for 8/10 targets (v3_score > v2_score)
3. **But 2026 pass_rate still 0%** for 9/10 targets

**Hypothesis:** The model's decision boundary still fails in 2026 because:
- The *relationship* between features and outcomes changes in extreme volatility
- A breakout at 0.45 ATRs in 2010-2023 meant something different than a breakout at 0.45 ATRs in 2026
- The label structure itself (TP/SL within fixed time windows) creates different risk/reward profiles when volatility is 5× higher
- Even with normalized features, the *joint distribution* of all features is different — the model sees combinations it never saw in training

## Best Target: y_1r4_180m

| Metric | v2 | v3 | Change |
|--------|:--:|:--:|:------:|
| Overall score | +0.188 | **+0.223** | +0.035 ✅ |
| Pass rate | 22.9% | **26.0%** | +3.1pp ✅ |
| Fail MLL | 4.1% | 3.8% | -0.3pp ✅ |
| 2025 pass | 30.5% | **34.7%** | +4.2pp ✅ |
| 2026 pass | 0.0% | 0.0% | ➡️ same |
| 2026 fail MLL | 0.0% | 0.0% | ➡️ same |
| 2026 median PnL | -$140 | -$356 | ❌ worse loss |
| Median end PnL | $683 | $399 | ❌ lower |

y_1r4_180m improves overall but still cannot pass Topstep in 2026.

## Key Stat Summary

| Metric | v2 | v3 |
|--------|:--:|:--:|
| Best score | +0.188 (y_1r4_180m) | **+0.223** (y_1r4_180m) |
| Targets > 0 score | 7/10 | **8/10** |
| Targets with 2026 pass | 1/10 (6.6% close60m) | 1/10 (4.9% y_1r2_180m) |
| 2025 best pass | 33.1% (y_1r4_60m) | **43.9%** (y_1r2_60m) |
| #1 feature | orb_range_atr_ratio | **price_vs_vwap_pct_abs** (25.3%) |

## Conclusion: Scale-Invariant Features Help, But Not Enough

**What worked:**
- Scale-invariant features (especially `price_vs_vwap_pct_abs`) are highly informative
- AB test shows 8/10 targets improved
- Feature distributions are stable across regimes (good OOD property)
- The new features dominate importance (4 of top 6 features)

**What didn't work:**
- 2026 pass_rate remains 0% for all targets except y_1r2_180m (4.9%)
- The model still fails OOD in 2026 despite normalized features
- y_1r4_close60m lost its 6.6% 2026 edge from v2

**Next direction needed:**
The problem is deeper than feature scaling. The model's *decision function* doesn't generalize to 2026. Potential approaches:
1. **Regime-conditional models** — separate high-vol and low-vol models
2. **Full retrain through 2025** — include 2024-2025 moderate volatility in training
3. **Alternative model architecture** — XGBoost with different regularization, or a simpler logistic regression with handcrafted features
4. **Rolling retrain** — retrain weekly/monthly on most recent data only
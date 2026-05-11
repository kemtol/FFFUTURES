# Objective Sweep v6 — Modular Feature Architecture

## Key Changes from v5

| Aspect | v5 | v6 |
|--------|:--:|:--:|
| Feature source | Hardcoded in script | **Modules from `data/Level_1_Features/modules/`** |
| Scale-invariant features | Inline `add_scale_invariant_features()` | **`scale_invariant_features` module** |
| Context features | From datamart (patched by `build_market_context.py`) | **`orb_context_features` module** |
| Adding new features | Modify `build_market_context.py` + rebuild datamart | **Create new module generator → run → re-sweep** |
| Feature list | `V2_FEATURES` + `V3_NEW_FEATURES` hardcoded | **Auto-derived: core + module columns** |

## Summary

| Metric | Value |
|--------|-------|
| Labels evaluated | 10 |
| Training data | 2025-11-30 (recent regime) |
| Calibration | 2025-07-01 → 2025-11-30 |
| Holdout | 2025-12-01+ |
| Param grid | 128 combinations |
| Simulator | Refined (CT trade day + $3 commission) |
| Feature modules | 4 |

## Ranked Results (v6 — best params per label)

| target | rr | score | pass_rate | fail_mll_rate | median_end_pnl | avg_trades | windows |
|---|---|---|---|---|---|---|---|
| y_1r4_240m | 4.000 | 0.590 | 59.0% | 0.0% | 2614.000 | 48.084 | 83 |
| y_1r2_60m | 2.000 | 0.494 | 49.4% | 0.0% | 2682.000 | 38.181 | 83 |
| y_1r2_180m | 2.000 | 0.434 | 43.4% | 0.0% | 2133.000 | 35.759 | 83 |
| y_1r2_120m | 2.000 | 0.386 | 38.6% | 0.0% | 1836.000 | 33.771 | 83 |
| y_1r2_close60m | 2.000 | 0.349 | 34.9% | 0.0% | 2106.000 | 39.181 | 83 |
| y_1r2_240m | 2.000 | 0.277 | 27.7% | 0.0% | 1836.000 | 36.361 | 83 |
| y_1r4_120m | 4.000 | 0.253 | 26.5% | 1.2% | 2584.000 | 55.265 | 83 |
| y_1r4_180m | 4.000 | 0.241 | 26.5% | 2.4% | 3084.000 | 55.843 | 83 |
| y_1r4_close60m | 4.000 | 0.241 | 37.3% | 13.3% | 2292.000 | 40.976 | 83 |
| y_1r4_60m | 4.000 | 0.120 | 30.1% | 18.1% | 2101.000 | 38.386 | 83 |

## Yearly Pass Rate by Label (v6)

| target | pass_2025 | pass_2026 | pass_rate |
|---|---|---|---|
| y_1r4_240m | 0.0% | 70.5% | 59.0% |
| y_1r2_60m | 100.0% | 60.7% | 49.4% |
| y_1r2_180m | 100.0% | 45.9% | 43.4% |
| y_1r2_120m | 100.0% | 45.9% | 38.6% |
| y_1r2_close60m | 100.0% | 34.4% | 34.9% |
| y_1r2_240m | 33.3% | 36.1% | 27.7% |
| y_1r4_120m | 0.0% | 34.4% | 26.5% |
| y_1r4_180m | 0.0% | 36.1% | 26.5% |
| y_1r4_close60m | 0.0% | 45.9% | 37.3% |
| y_1r4_60m | 0.0% | 41.0% | 30.1% |

## Yearly Fail MLL Rate by Label (v6)

| target | fail_mll_2025 | fail_mll_2026 | fail_mll_rate |
|---|---|---|---|
| y_1r4_240m | 0.0% | 0.0% | 0.0% |
| y_1r2_60m | 0.0% | 0.0% | 0.0% |
| y_1r2_180m | 0.0% | 0.0% | 0.0% |
| y_1r2_120m | 0.0% | 0.0% | 0.0% |
| y_1r2_close60m | 0.0% | 0.0% | 0.0% |
| y_1r2_240m | 0.0% | 0.0% | 0.0% |
| y_1r4_120m | 0.0% | 0.0% | 1.2% |
| y_1r4_180m | 0.0% | 3.3% | 2.4% |
| y_1r4_close60m | 100.0% | 0.0% | 13.3% |
| y_1r4_60m | 100.0% | 3.3% | 18.1% |

## Yearly Median PnL by Label (v6)

| target | pnl_2025 | pnl_2026 |
|---|---|---|
| y_1r4_240m | $+1702 | $+2777 |
| y_1r2_60m | $+2979 | $+3789 |
| y_1r2_180m | $+2853 | $+2421 |
| y_1r2_120m | $+2538 | $+2025 |
| y_1r2_close60m | $+3447 | $+2007 |
| y_1r2_240m | $+2079 | $+2178 |
| y_1r4_120m | $+3496 | $+2541 |
| y_1r4_180m | $+2569 | $+3026 |
| y_1r4_close60m | $-1283 | $+2586 |
| y_1r4_60m | $-1180 | $+2586 |

## Best Params per Label

| target | target | rev_q | cont_q | rev_adx_min | cont_adx_max | risk_per_r_usd | daily_profit_cap |
|---|---|---|---|---|---|---|---|
| y_1r4_240m | y_1r4_240m | 0.750 | 0.750 | 30 | 30 | 100.000 | 1400.000 |
| y_1r2_60m | y_1r2_60m | 0.750 | 0.600 | 30 | 100 | 150.000 | 1400.000 |
| y_1r2_180m | y_1r2_180m | 0.750 | 0.600 | 30 | 100 | 150.000 | 1400.000 |
| y_1r2_120m | y_1r2_120m | 0.750 | 0.600 | 30 | 100 | 150.000 | 1400.000 |
| y_1r2_close60m | y_1r2_close60m | 0.750 | 0.600 | 30 | 100 | 150.000 | 1400.000 |
| y_1r2_240m | y_1r2_240m | 0.750 | 0.600 | 30 | 100 | 150.000 | 1400.000 |
| y_1r4_120m | y_1r4_120m | 0.750 | 0.750 | 30 | 30 | 100.000 | 0.000 |
| y_1r4_180m | y_1r4_180m | 0.600 | 0.750 | 40 | 30 | 100.000 | 0.000 |
| y_1r4_close60m | y_1r4_close60m | 0.750 | 0.750 | 30 | 30 | 100.000 | 1400.000 |
| y_1r4_60m | y_1r4_60m | 0.750 | 0.750 | 30 | 30 | 100.000 | 1400.000 |

## Feature Modules Used

- `orb_context_features.parquet`: 7 features — `orb_range_atr_ratio`, `day_of_week`, `time_in_session_min`, `vwap_at_breakout`, `price_vs_vwap_pct`, `adx_14_15m`, `ema_slope_1h`
- `pre_breakout_profile_features.parquet`: 5 features — `pre_bo_compression_ratio`, `pre_bo_drift_atr`, `pre_bo_inside_bar_flag`, `pre_bo_last_candle_range_ratio`, `pre_bo_bullish_ratio`
- `scale_invariant_features.parquet`: 7 features — `breakout_strength_atr_ratio`, `atr14_sq`, `breakout_strength_sq`, `orb_range_sq`, `price_vs_vwap_pct_abs`, `adx_50_flag`, `breakout_strength_vs_orb`
- `volatility_normalized_features.parquet`: 5 features — `atr14_percentile_20d`, `atr14_zscore_20d`, `breakout_strength_percentile_20d`, `breakout_strength_zscore_10d`, `orb_range_percentile_20d`

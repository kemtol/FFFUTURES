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
| Feature modules | 8 |

## Ranked Results (v6 — best params per label)

| target | rr | score | pass_rate | fail_mll_rate | median_end_pnl | avg_trades | windows |
|---|---|---|---|---|---|---|---|
| y_1r2_180m | 2.000 | 0.651 | 65.1% | 0.0% | 3059.000 | 35.880 | 83 |
| y_1r4_180m | 4.000 | 0.614 | 61.4% | 0.0% | 3144.000 | 50.205 | 83 |
| y_1r2_60m | 2.000 | 0.590 | 59.0% | 0.0% | 2928.000 | 31.398 | 83 |
| y_1r2_close60m | 2.000 | 0.554 | 55.4% | 0.0% | 2698.000 | 33.795 | 83 |
| y_1r2_240m | 2.000 | 0.554 | 55.4% | 0.0% | 3059.000 | 29.807 | 83 |
| y_1r4_close60m | 4.000 | 0.470 | 62.7% | 15.7% | 3156.000 | 42.361 | 83 |
| y_1r4_60m | 4.000 | 0.446 | 44.6% | 0.0% | 2631.000 | 38.867 | 83 |
| y_1r4_240m | 4.000 | 0.446 | 44.6% | 0.0% | 2468.000 | 52.663 | 83 |
| y_1r2_120m | 2.000 | 0.410 | 65.1% | 24.1% | 3642.000 | 27.458 | 83 |
| y_1r4_120m | 4.000 | 0.349 | 41.0% | 6.0% | 1983.000 | 43.024 | 83 |

## Yearly Pass Rate by Label (v6)

| target | pass_2025 | pass_2026 | pass_rate |
|---|---|---|---|
| y_1r2_180m | 33.3% | 75.4% | 65.1% |
| y_1r4_180m | 0.0% | 77.0% | 61.4% |
| y_1r2_60m | 66.7% | 72.1% | 59.0% |
| y_1r2_close60m | 33.3% | 73.8% | 55.4% |
| y_1r2_240m | 66.7% | 72.1% | 55.4% |
| y_1r4_close60m | 0.0% | 72.1% | 62.7% |
| y_1r4_60m | 0.0% | 60.7% | 44.6% |
| y_1r4_240m | 0.0% | 57.4% | 44.6% |
| y_1r2_120m | 100.0% | 60.7% | 65.1% |
| y_1r4_120m | 0.0% | 55.7% | 41.0% |

## Yearly Fail MLL Rate by Label (v6)

| target | fail_mll_2025 | fail_mll_2026 | fail_mll_rate |
|---|---|---|---|
| y_1r2_180m | 0.0% | 0.0% | 0.0% |
| y_1r4_180m | 0.0% | 0.0% | 0.0% |
| y_1r2_60m | 0.0% | 0.0% | 0.0% |
| y_1r2_close60m | 0.0% | 0.0% | 0.0% |
| y_1r2_240m | 0.0% | 0.0% | 0.0% |
| y_1r4_close60m | 100.0% | 0.0% | 15.7% |
| y_1r4_60m | 0.0% | 0.0% | 0.0% |
| y_1r4_240m | 0.0% | 0.0% | 0.0% |
| y_1r2_120m | 0.0% | 32.8% | 24.1% |
| y_1r4_120m | 0.0% | 8.2% | 6.0% |

## Yearly Median PnL by Label (v6)

| target | pnl_2025 | pnl_2026 |
|---|---|---|
| y_1r2_180m | $+2671 | $+4110 |
| y_1r4_180m | $+908 | $+3453 |
| y_1r2_60m | $+2662 | $+3722 |
| y_1r2_close60m | $+2247 | $+3722 |
| y_1r2_240m | $+2865 | $+4128 |
| y_1r4_close60m | $-825 | $+3309 |
| y_1r4_60m | $-298 | $+3159 |
| y_1r4_240m | $+511 | $+2880 |
| y_1r2_120m | $+3633 | $+3651 |
| y_1r4_120m | $+114 | $+2410 |

## Best Params per Label

| target | target | rev_q | cont_q | rev_adx_min | cont_adx_max | risk_per_r_usd | daily_profit_cap |
|---|---|---|---|---|---|---|---|
| y_1r2_180m | y_1r2_180m | 0.750 | 0.600 | 30 | 100 | 200.000 | 1400.000 |
| y_1r4_180m | y_1r4_180m | 0.750 | 0.750 | 30 | 30 | 100.000 | 1400.000 |
| y_1r2_60m | y_1r2_60m | 0.750 | 0.600 | 30 | 100 | 200.000 | 1400.000 |
| y_1r2_close60m | y_1r2_close60m | 0.750 | 0.600 | 30 | 100 | 200.000 | 1400.000 |
| y_1r2_240m | y_1r2_240m | 0.750 | 0.600 | 30 | 100 | 200.000 | 1400.000 |
| y_1r4_close60m | y_1r4_close60m | 0.750 | 0.600 | 30 | 30 | 150.000 | 1400.000 |
| y_1r4_60m | y_1r4_60m | 0.750 | 0.750 | 30 | 30 | 100.000 | 1400.000 |
| y_1r4_240m | y_1r4_240m | 0.600 | 0.750 | 30 | 30 | 100.000 | 1400.000 |
| y_1r2_120m | y_1r2_120m | 0.750 | 0.600 | 30 | 100 | 250.000 | 1400.000 |
| y_1r4_120m | y_1r4_120m | 0.750 | 0.750 | 30 | 30 | 100.000 | 1400.000 |

## Feature Modules Used

- `interaction_features.parquet`: 5 features — `int_atr14_x_adx`, `int_breakout_strength_x_range`, `int_vwap_distance_x_atr14`, `int_adx_x_orb_range`, `int_breakout_strength_x_session`
- `macro_features.parquet`: 4 features — `mac_spx_regime`, `mac_dxy_trend`, `mac_us10y_change`, `mac_oil_volatility`
- `orb_context_features.parquet`: 7 features — `orb_range_atr_ratio`, `day_of_week`, `time_in_session_min`, `vwap_at_breakout`, `price_vs_vwap_pct`, `adx_14_15m`, `ema_slope_1h`
- `pre_breakout_profile_features.parquet`: 5 features — `pre_bo_compression_ratio`, `pre_bo_drift_atr`, `pre_bo_inside_bar_flag`, `pre_bo_last_candle_range_ratio`, `pre_bo_bullish_ratio`
- `regime_features.parquet`: 3 features — `regime_state`, `volatility_zscore`, `efficiency_ratio`
- `scale_invariant_features.parquet`: 7 features — `breakout_strength_atr_ratio`, `atr14_sq`, `breakout_strength_sq`, `orb_range_sq`, `price_vs_vwap_pct_abs`, `adx_50_flag`, `breakout_strength_vs_orb`
- `session_momentum_features.parquet`: 4 features — `sm_first_30m_range`, `sm_first_30m_direction`, `sm_pre_breakout_volume_ratio`, `sm_pre_breakout_volume_z`
- `volatility_normalized_features.parquet`: 5 features — `atr14_percentile_20d`, `atr14_zscore_20d`, `breakout_strength_percentile_20d`, `breakout_strength_zscore_10d`, `orb_range_percentile_20d`

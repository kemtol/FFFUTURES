# Model Report — ORB_v1.0

Generated: 2026-04-26 07:24

## Summary

| Metric | Value |
|--------|-------|
| Version | 1.0 |
| Target | `y_1r2_120m` |
| Side | reversal only |
| Training rows | 28,884 |
| CV folds | 23 |
| Holdout locked from | 2024-01-01 |
| Sample weighting | Exponential decay, half-life=2y |

## CV Performance (all folds)

| Metric | Value |
|--------|-------|
| Mean AUC | 0.6234 |
| Mean win rate (top 40%) | 0.518 |
| Mean exp net (top 40%) | 0.485R |

## CV Performance (2020+ folds only — recent regime)

| Metric | Value |
|--------|-------|
| Mean AUC | 0.5838 |
| Mean exp net (top 40%) | 0.153R |

## Walk-Forward Fold Results

| Fold | Period | AUC | WR All | WR Top40% | Exp Net |
|------|--------|-----|--------|-----------|---------|
|  1 | 2012-10-04 → 2013-04-03 | 0.6030 | 0.401 | 0.472 | 0.346R |
|  2 | 2013-04-04 → 2013-10-03 | 0.5860 | 0.350 | 0.432 | 0.226R |
|  3 | 2013-10-04 → 2014-04-03 | 0.5097 | 0.378 | 0.380 | 0.070R |
|  4 | 2014-04-04 → 2014-10-03 | 0.5994 | 0.347 | 0.439 | 0.246R |
|  5 | 2014-10-06 → 2015-04-02 | 0.5955 | 0.399 | 0.470 | 0.339R |
|  6 | 2015-04-06 → 2015-10-02 | 0.5877 | 0.394 | 0.455 | 0.294R |
|  7 | 2015-10-05 → 2016-04-01 | 0.5662 | 0.391 | 0.432 | 0.227R |
|  8 | 2016-04-04 → 2016-10-03 | 0.5739 | 0.473 | 0.548 | 0.573R |
|  9 | 2016-10-04 → 2017-04-03 | 0.6167 | 0.529 | 0.655 | 0.895R |
| 10 | 2017-04-04 → 2017-10-03 | 0.7947 | 0.706 | 0.896 | 1.619R |
| 11 | 2017-10-04 → 2018-04-03 | 0.8470 | 0.651 | 0.925 | 1.704R |
| 12 | 2018-04-04 → 2018-10-03 | 0.8754 | 0.697 | 0.978 | 1.865R |
| 13 | 2018-10-04 → 2019-04-03 | 0.6947 | 0.499 | 0.636 | 0.839R |
| 14 | 2019-04-04 → 2019-10-03 | 0.6475 | 0.398 | 0.520 | 0.489R |
| 15 | 2019-10-04 → 2020-04-03 | 0.5707 | 0.367 | 0.418 | 0.184R |
| 16 | 2020-04-06 → 2020-10-02 | 0.5786 | 0.312 | 0.350 | -0.021R |
| 17 | 2020-10-05 → 2021-04-01 | 0.5497 | 0.331 | 0.369 | 0.038R |
| 18 | 2021-04-05 → 2021-10-01 | 0.5622 | 0.348 | 0.397 | 0.120R |
| 19 | 2021-10-04 → 2022-04-01 | 0.5277 | 0.364 | 0.377 | 0.061R |
| 20 | 2022-04-04 → 2022-10-03 | 0.6059 | 0.387 | 0.484 | 0.381R |
| 21 | 2022-10-04 → 2023-04-03 | 0.5961 | 0.316 | 0.401 | 0.134R |
| 22 | 2023-04-04 → 2023-10-03 | 0.6187 | 0.370 | 0.452 | 0.285R |
| 23 | 2023-10-04 → 2023-12-29 | 0.6315 | 0.338 | 0.433 | 0.230R |

## Feature Importance (Gain)

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `price_vs_vwap_pct` | 9286 |
| 2 | `orb_range_atr_ratio` | 6895 |
| 3 | `atr14_at_entry` | 6745 |
| 4 | `adx_14_15m` | 6683 |
| 5 | `breakout_strength` | 3850 |
| 6 | `time_in_session_min` | 3490 |
| 7 | `day_of_week` | 1512 |
| 8 | `session` | 881 |

## Config

```json
{
  "lgbm_params": {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "n_jobs": -1
  },
  "features": [
    "orb_range_atr_ratio",
    "breakout_strength",
    "atr14_at_entry",
    "price_vs_vwap_pct",
    "adx_14_15m",
    "ema_slope_1h",
    "day_of_week",
    "time_in_session_min",
    "orb_tf",
    "session",
    "breakout_side"
  ]
}
```

## Notes

- Baseline reversal win rate (no filter): ~40% gross, ~33% net of costs
- Top 40% threshold targets setups where model assigns highest probability
- 2017–2018 regime anomaly (wr ~65%) is downweighted via exponential decay
- Holdout 2024+ not evaluated — reserved for final model validation

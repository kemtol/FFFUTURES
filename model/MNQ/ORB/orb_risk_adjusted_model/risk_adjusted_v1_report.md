# MNQ ORB Risk-Adjusted V1 Report

Created: `2026-05-29T03:45:12.475358+00:00`

## Dataset

- Rows: `2559`
- Features: `33`
- Train/validation/holdout: `{'holdout': 140, 'train': 2068, 'validation': 351}`

## Models

### Target `success_2r`

- `logistic` validation AUC: `0.7223587223587223` Brier: `0.21424141702807395`
- `logistic` holdout AUC: `0.7162356321839081` Brier: `0.1967665346150104`
- `logistic` artifact: `/home/kemal/futures/model/MNQ/ORB/orb_risk_adjusted_model/risk_adjusted_v1_success_2r_logistic.joblib`
- `lgbm_shallow` validation AUC: `0.6225122850122851` Brier: `0.12873482837052236`
- `lgbm_shallow` holdout AUC: `0.605962643678161` Brier: `0.13985900101532694`
- `lgbm_shallow` artifact: `/home/kemal/futures/model/MNQ/ORB/orb_risk_adjusted_model/risk_adjusted_v1_success_2r_lgbm_shallow.joblib`

### Target `positive_eod`

- `logistic` validation AUC: `0.5342312008978676` Brier: `0.25097423685525533`
- `logistic` holdout AUC: `0.541875` Brier: `0.24860229063551773`
- `logistic` artifact: `/home/kemal/futures/model/MNQ/ORB/orb_risk_adjusted_model/risk_adjusted_v1_positive_eod_logistic.joblib`
- `lgbm_shallow` validation AUC: `0.5466594045025418` Brier: `0.2443052396001667`
- `lgbm_shallow` holdout AUC: `0.5497916666666667` Brier: `0.2423738780596859`
- `lgbm_shallow` artifact: `/home/kemal/futures/model/MNQ/ORB/orb_risk_adjusted_model/risk_adjusted_v1_positive_eod_lgbm_shallow.joblib`

## Caveats

- V1 is a probability-separation test, not live approval.
- Holdout has only 140 breakout rows, so threshold decisions need walk-forward and Topstep overlay checks.
- Labels are outcome labels and must not be used as features.

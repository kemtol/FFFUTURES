# Meta-v8 CONS VWAP Research

Research-only workspace for a new Conservative ML candidate. This does not
touch the live V8 router, live Meta-v7 model, or systemd services.

## Goal

Test whether VWAP context can sharpen the current CONS path:

- Baseline live CONS model: `model/SUPER_STRUCTURE/meta_v7/inference_model.txt`
- Baseline live CONS config: `model/SUPER_STRUCTURE/meta_v7/inference_config_refined.json`
- Baseline datamart: `data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet`

## Candidate Feature Families

Baseline Meta-v7 features:

- `entry_adx`
- `cci_abs`
- `st_gap_ratio`
- `efficiency_ratio`
- `volatility_zscore`
- `session_cluster`

VWAP candidate features:

- `dist_to_ct_vwap_atr`
- `vwap_side_aligned`
- `ct_vwap_slope_20_atr`
- `vwap_deviation_z_50`

## Hard Isolation Rules

- Do not edit `pipeline/live/inference_router.py`.
- Do not edit `model/SUPER_STRUCTURE/meta_v7/*`.
- Do not write to `model/SUPER_STRUCTURE/meta_v8` unless explicitly promoting.
- Candidate outputs must use `meta_v8_cons_vwap` names.
- Promotion requires walk-forward pass and router/live parity verification.

## Intended Flow

1. Build candidate datamart:

   ```bash
   python3 pipeline/super_structure_ml/experiments/meta_v8_cons_vwap/build_l1_vwap.py --dry-run
   python3 pipeline/super_structure_ml/experiments/meta_v8_cons_vwap/build_l1_vwap.py
   python3 pipeline/super_structure_ml/experiments/meta_v8_cons_vwap/build_features.py --dry-run
   python3 pipeline/super_structure_ml/experiments/meta_v8_cons_vwap/build_features.py
   ```

2. Train candidate model into an isolated output directory:

   ```bash
   python3 pipeline/super_structure_ml/experiments/meta_v8_cons_vwap/train_candidate.py --dry-run
   python3 pipeline/super_structure_ml/experiments/meta_v8_cons_vwap/train_candidate.py
   ```

3. Simulate 7d/14d/30d/90d against Meta-v7 baseline.
4. Only after it wins, create a separate live builder parity test.

## Hypertuning

Hypertuning uses a stricter split than the initial candidate trainer:

- Train: 2023-2024
- Tune hyperparameters and thresholds: 2025
- Report only: 2026

```bash
python3 pipeline/super_structure_ml/experiments/meta_v8_cons_vwap/hypertune_candidate.py --limit 5 --dry-run
python3 pipeline/super_structure_ml/experiments/meta_v8_cons_vwap/hypertune_candidate.py
```

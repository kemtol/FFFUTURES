# Super Structure Model Artifacts

Versioned model outputs for Super Structure research.

## Live (as of 2026-05-14)

V8 router uses **Meta-v7 Refined** as Conservative brain + **v1.12 mechanical**
as Aggressive filter (no ML brain).

- `meta_v7/inference_model.txt` — Conservative LightGBM brain (6 features).
- `meta_v7/inference_config_refined.json` — dynamic thresholds per
  `session_cluster`: `{0:0.50, 1:0.50, 2:0.45}`. Source of truth for
  `pipeline/live/inference_router.py:_threshold_map`.
- `meta_v7/inference_config.json` — older static threshold variant; not used
  by V8 router.
- `meta_v7/reports/` — Monte Carlo + walk-forward reports for CONS solo
  (16.94% prob_of_ruin).
- `meta_v7/evolution/` — Meta-v7 variant configs (v7a aggressive, v7b strict
  sniper, v7c refined final).

Aggressive (mechanical) brain artifacts are **not stored as model files** —
the rule is parameterized in code:
- Detector: `pipeline/live/pullback_detector.py` (mirrors `build_training_datamart_v1_12.py`).
- Filter: `risk_pts <= 12` in `pipeline/live/inference_router.py`.
- Datamart: `data/Level_2_Datamart/super_structure_ml/v1_12_training_datamart.parquet`.

## Legacy

- `SMART_1/` — pre-V8 dual-ML stack (regime_dispatcher + cons_brain +
  aggr_brain). `conservative_brain.txt` is byte-identical to
  `meta_v7/inference_model.txt`. **Keep `aggressive_brain.txt`** for rollback
  path (toggle `USE_V8_ROUTER=False`).
- `meta_v1/` through `meta_v6/` — historical baselines, see
  `_ARCH/model_backups/super_structure_legacy/`.
- `simulation-compare/` — windowed sim ledgers for direct comparison
  (`TEMP_SIM_*` = older variants; `SIM_CONS_ML_AGGR_MECH_*` = V8 validated
  reference; `REPLAY_ROUTER_*` = sync verifier output).

# Inference Chain Health — 2026-05-16 07:33 WIB

**Status: ✅ PASS**

Baselines captured: `2026-05-14T08:52:52.445378+00:00`

## B. Model Artifact Integrity
Severity: ✅ PASS

| File | Severity | Note |
| --- | --- | --- |
| `model/SUPER_STRUCTURE/meta_v7/inference_model.txt` | ✅ PASS |  |
| `model/SUPER_STRUCTURE/meta_v7/inference_config_refined.json` | ✅ PASS |  |
| `model/SUPER_STRUCTURE/SMART_1/conservative_brain.txt` | ✅ PASS |  |
| _inference_model_eq_conservative_brain_ | ✅ PASS | byte-identical |
| _refined_config_structure_ | ✅ PASS |  |
| _lightgbm_loadable_ | ✅ PASS |  |

Refined thresholds: `{'0': 0.5, '1': 0.5, '2': 0.45}`

## C. L0 Raw ↔ L0 Live Sync
Severity: ✅ PASS

- Raw rows: `0`, Live rows: `145` (no overlap to compare)
- Note: raw DB stale (batch-ingested, expected)

## D. Datamart Anchor Sterility
Severity: ✅ PASS

| File | Anchor | Actual | Severity | Note |
| --- | ---: | ---: | --- | --- |
| `data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet` | 3643 | 3643 | ✅ PASS |  |
| `data/Level_2_Datamart/super_structure_ml/v1_12_training_datamart.parquet` | 1471 | 1471 | ✅ PASS |  |

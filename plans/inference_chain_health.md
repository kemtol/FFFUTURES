# Plan: Inference Chain Health Check (Phase 2)

## Context

Phase 1 (`plans/super_structure_data_health.md`) covers `topstepx_buffer.db`
— the only data input V8 router actually consumes at inference time. But the
"data feeding inference" chain is broader. The inputs that V8 *trusts* but
could silently drift:

```
                  TopstepX WS
                       │
                       ▼
        topstepx_buffer.db ◄────── Phase 1 covers
                       │
                       ▼
              SMARTFeatureBuilder       (recompute in-memory)
                       │
                       ▼
              InferenceRouter
                  │           │
        ┌─────────┴───┐  ┌────┴──────┐
        ▼             ▼  ▼          ▼
  meta_v7/        SMART_1/  inference_config_refined.json
  inference_      conservative_     │
  model.txt       brain.txt    {"thresholds":{...}}
  (Phase 2 B) ◄── (Phase 2 B) ◄── (Phase 2 B)

  v3_final_training.parquet   v1_12_training_datamart.parquet
       │                            │
       └──────── (Phase 2 D) ───────┘
       (sterile training/sim sources — sterility = sim reproducibility)

  data/Level_0_Raw/MGC_{1m,5m,15m}.db
                       │
                       ▼  (Phase 2 C)
              comparison vs topstepx_buffer.db on overlap
              (research integrity, not live trading)
```

Per user decision, Phase 2 scope = **B + C + D**:

- **B. Model artifact integrity** — detect silent model/config swap.
- **C. L0 Raw ↔ L0 Live sync** — detect raw/live drift (research integrity).
- **D. Datamart anchor sterility** — detect accidental overwrite.

Excluded from Phase 2:
- A. Upstream feed health (TopstepX WS, JWT, backfill rate) — covered loosely
  by Phase 1 freshness check.
- E. Feature parity (live vs training feature builder) — own phase, complex.

## Design Decisions

| | Choice | Why |
|---|---|---|
| Cadence | **Every 1 hour** | Phase 2 targets are sterile or slow-changing. 15min would be noise; 1h still catches drift within trading day. |
| Severity action | Telegram alert only, no auto-halt | Consistent with Phase 1 D2. |
| Baselines | Capture on first run, store in `data/Live/health/inference_chain_baselines.json` | Subsequent runs compare. Manual reset = delete file. |
| Output | JSON + Markdown + Telegram (always push) | Match Phase 1 pattern. |

## Files

**Create:**
- `pipeline/live/inference_chain_health_check.py` — main script
- `pipeline/run/super_structure_inference_chain_health.service`
- `pipeline/run/super_structure_inference_chain_health.timer`
- `data/Live/health/inference_chain_baselines.json` (auto-generated first run)
- `data/Live/health/inference_chain_{date}_{tz}.{json,md}` (per run)

**Read-only:**
- `model/SUPER_STRUCTURE/meta_v7/inference_model.txt`
- `model/SUPER_STRUCTURE/meta_v7/inference_config_refined.json`
- `model/SUPER_STRUCTURE/SMART_1/conservative_brain.txt`
- `data/Level_0_Raw/MGC_1m.db`
- `data/Live/topstepx_buffer.db`
- `data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet`
- `data/Level_2_Datamart/super_structure_ml/v1_12_training_datamart.parquet`

## Checks (3 categories)

### B. Model Artifact Integrity

For each of `inference_model.txt`, `conservative_brain.txt`,
`inference_config_refined.json`:

- File exists.
- sha256 hash computed.
- Compare to baseline; CRITICAL if drift.

Additional:
- `inference_model.txt` byte-identical to `SMART_1/conservative_brain.txt`
  (live wires meta_v7 path; SMART_1 path = legacy fallback).
- `inference_config_refined.json` parses, has `thresholds` dict with keys
  `0`, `1`, `2`, values float in [0.0, 1.0]. CRITICAL if invalid.
- LightGBM Booster loadable (sanity); CRITICAL if load fails.

### C. L0 Raw ↔ L0 Live Sync

Read last 6 hours of overlap window from BOTH `MGC_1m.db` and
`topstepx_buffer.db`:

- Both DBs have data in overlap.
- Merge on `timestamp_utc`; count matching rows.
- For matched rows, compare OHLC values; MGC tick = 0.10, so flag drift
  > 0.5 (5 ticks).
- Report: row count alignment, mean abs delta per column, max delta.
- WARN if row count mismatch > 5% or any column drift > 0.5 in > 1% of rows.
- CRITICAL if > 30% of rows are mismatched (raw stale by > 12h is acceptable;
  raw should generally lag live by hours, not days — flag heavy stale).

### D. Datamart Anchor Sterility

Two sterile datamarts:

| File | Anchor rows | Why sterile |
|---|---:|---|
| `v3_final_training.parquet` | **3643** | CONS source-of-truth for Meta-v7 training + sim |
| `v1_12_training_datamart.parquet` | **1471** | AGGR mechanical source for v1.12 + sim |

Per file:
- Row count == anchor → PASS; otherwise CRITICAL.
- sha256 vs baseline → PASS or CRITICAL.
- File mtime drift (warn-only if changed but content same).

If user intentionally rebuilds datamart (rare), they manually delete
`inference_chain_baselines.json` to reset.

## Baselines File

`data/Live/health/inference_chain_baselines.json`:

```json
{
  "captured_at_utc": "2026-05-14T08:00:00+00:00",
  "captured_by": "first run",
  "model_artifacts": {
    "model/SUPER_STRUCTURE/meta_v7/inference_model.txt": {
      "sha256": "...",
      "size_bytes": 341063
    },
    "model/SUPER_STRUCTURE/meta_v7/inference_config_refined.json": {
      "sha256": "...",
      "size_bytes": 419
    },
    "model/SUPER_STRUCTURE/SMART_1/conservative_brain.txt": {
      "sha256": "...",
      "size_bytes": 341063
    }
  },
  "datamart_anchors": {
    "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet": {
      "rows": 3643,
      "sha256": "..."
    },
    "data/Level_2_Datamart/super_structure_ml/v1_12_training_datamart.parquet": {
      "rows": 1471,
      "sha256": "..."
    }
  }
}
```

First run: if file missing → capture + tag every check as "baseline captured,
no comparison yet". Subsequent runs: compare and flag drift.

## Severity Matrix

| Check | PASS | WARN | CRITICAL |
|---|---|---|---|
| Model file exists | exists | — | missing |
| Model sha256 vs baseline | match | — | drift |
| inference_model == conservative_brain | byte-identical | — | not equal |
| Refined config JSON valid | valid | — | parse error |
| Refined thresholds in [0,1] | yes | — | out of range |
| LightGBM loadable | OK | — | load fails |
| L0/Live overlap row count | within 5% | 5-30% mismatch | > 30% mismatch |
| L0/Live OHLC drift | < 0.5 in < 1% rows | drift in 1-5% | drift in > 5% |
| Datamart row count | == anchor | — | != anchor |
| Datamart sha256 | match baseline | mtime changed, sha same | sha drift |

## Implementation Skeleton

```python
# pipeline/live/inference_chain_health_check.py
import argparse, hashlib, json, sqlite3
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent.parent
BASELINES = ROOT / "data/Live/health/inference_chain_baselines.json"
TELEGRAM_ENV_PATH = ROOT / "data/Live/telegram.env"

MODEL_PATHS = [
    ROOT / "model/SUPER_STRUCTURE/meta_v7/inference_model.txt",
    ROOT / "model/SUPER_STRUCTURE/meta_v7/inference_config_refined.json",
    ROOT / "model/SUPER_STRUCTURE/SMART_1/conservative_brain.txt",
]
DATAMART_ANCHORS = {
    "v3_final_training.parquet": 3643,
    "v1_12_training_datamart.parquet": 1471,
}

def sha256_file(path: Path) -> str: ...
def load_baselines() -> dict: ...
def save_baselines(d: dict) -> None: ...

def check_model_artifact(baselines): ...
def check_l0_live_sync(): ...
def check_datamart_anchors(baselines): ...

def main(): ...
```

## systemd Wiring

```ini
# pipeline/run/super_structure_inference_chain_health.service
[Unit]
Description=Super Structure Inference Chain Health (model + L0 sync + datamart anchors)
After=network.target

[Service]
Type=oneshot
WorkingDirectory=/home/kemal/futures
ExecStart=/usr/bin/python3 -u /home/kemal/futures/pipeline/live/inference_chain_health_check.py --telegram always
Environment=PYTHONUNBUFFERED=1
StandardOutput=append:/home/kemal/futures/data/Live/super_structure_inference_chain_health.log
StandardError=append:/home/kemal/futures/data/Live/super_structure_inference_chain_health.log

[Install]
WantedBy=default.target
```

```ini
# pipeline/run/super_structure_inference_chain_health.timer
[Unit]
Description=Run inference chain health check every hour

[Timer]
OnBootSec=5min
OnUnitActiveSec=1h
Persistent=true
Unit=super_structure_inference_chain_health.service

[Install]
WantedBy=timers.target
```

## Verification

1. Syntax + dry run with `--telegram never`. Expect baseline capture
   message on first run.
2. Second run: all checks PASS.
3. Synthetic drift test:
   ```bash
   # Modify refined config (will trigger CRITICAL)
   cp inference_config_refined.json inference_config_refined.json.bak
   echo '{"foo":"bar"}' > inference_config_refined.json
   python3 pipeline/live/inference_chain_health_check.py --telegram never
   # Expect CRITICAL
   mv inference_config_refined.json.bak inference_config_refined.json
   ```
4. Install timer, verify next firing scheduled, check log after first
   timer run.

## Out of Scope (Phase 3 candidates)

- **A. Upstream feed daemon health** — TopstepX WS daemon active/error rate,
  JWT token expiry, yfinance backfill frequency.
- **E. Feature parity** — replay live `smart_features.build_smart_features()`
  vs pre-computed features in v3 datamart for sample timestamps. Tolerance-
  based drift detection.
- **Auto-halt on Phase 2 CRITICAL** — currently alert-only per design. After
  Phase 1+2 noise pattern is known, decide whether model-drift or sterility-
  break should auto-halt.

## Risk & Rollback

- All checks read-only. Worst case: Telegram noise.
- Rollback: `systemctl --user disable --now super_structure_inference_chain_health.timer`.
- Baselines file regeneratable: `rm` it and next run captures fresh.

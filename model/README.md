# Model Registry

Central registry for model and strategy artifacts.

Use this file first when asking:

```text
model mana?
yang dipakai yang mana?
mana yang live, research, atau legacy?
```

## Current Decision Table

| Namespace | Model / Strategy | Instrument | Role | Status | Primary Folder |
| --- | --- | --- | --- | --- | --- |
| MNQ ORB | `rule_based_15m_long_tp2r_eod` | MNQ | Current research baseline | Best current MNQ baseline, not live-ready | `model/MNQ/ORB/rule_based_15m_long_tp2r_eod/` |
| MNQ ORB | `orb_risk_adjusted_model` | MNQ | ML risk-sizing overlay | Research only, not used for live trading | `model/MNQ/ORB/orb_risk_adjusted_model/` |
| Super Structure | `meta_v7` refined CONS + v1.12 mechanical AGGR | MGC | Live router components | Used by live Super Structure router | `model/SUPER_STRUCTURE/meta_v7/` |
| Super Structure | `SMART_1` | MGC | Legacy rollback/research | Not current default | `model/SUPER_STRUCTURE/SMART_1/` |

## Current MNQ Answer

Current named MNQ baseline:

```text
rule_based_15m_long_tp2r_eod
```

Definition:

```text
MNQ, NY session, M1 execution,
15m opening range, long only,
first M1 close above OR high,
entry next M1 open,
TP 2R or 15:00 NY time exit,
risk $500, no normal strategic SL.
```

This is **not live-ready** yet. It is the control candidate that every ML or
rule variant must beat.

Main model card:

```text
model/MNQ/ORB/
```

Canonical data artifacts:

```text
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/
```

## Current MGC Live Answer

Super Structure live is not using MNQ.

Current MGC live router:

```text
Meta-v7 Refined Conservative ML
plus v1.12 Aggressive mechanical risk filter
```

Model/config source:

```text
model/SUPER_STRUCTURE/meta_v7/inference_model.txt
model/SUPER_STRUCTURE/meta_v7/inference_config_refined.json
```

Live implementation:

```text
pipeline/live/inference_router.py
pipeline/live/super_structure.py
```

## Tracking Policy

`model/` is tracked by Git.

Tracked:

- model cards and README files
- configs and manifests
- metrics and reports
- small `.joblib` model files
- small `.txt` model files
- feature importance CSVs

Ignored globally:

- `*.parquet`
- `*.png`
- database files
- pycache files

If a small canonical parquet is truly part of a model package, force-add it
explicitly and document why.


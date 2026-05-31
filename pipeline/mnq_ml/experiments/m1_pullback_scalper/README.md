# MNQ M1 Pullback Scalper

Research-only scaffold for a standalone Micro Nasdaq M1 model.

## Intent

- Instrument: MNQ / Micro Nasdaq futures.
- Timeframe: M1 first.
- Core idea: trend continuation pullback with VWAP/regime confluence.
- Entry timing: signal on closed M1 bar, executable at next M1 open.
- Exit: condition-aware TP/SL simulation, with slippage and commission modeled.
- Goal: positive expectancy with enough frequency for a scalping sleeve.

## Isolation

- Do not edit `pipeline/live`.
- Do not edit `model/SUPER_STRUCTURE`.
- Do not read Gold/MGC datamarts as a fallback.
- Keep all MNQ outputs under `data/Level_1_Features/mnq`,
  `data/Level_2_Datamart/mnq`, and `model/MNQ`.

## First Flow

The scaffold uses `data/Level_0_Raw/MNQ_1m.duckdb`, built from the Databento
NDJSON backfill. Rebuild it with:

```bash
python3 pipeline/mnq_ml/build_l0_duckdb_from_databento.py --force
```

```bash
python3 pipeline/mnq_ml/experiments/m1_pullback_scalper/gate_training_data.py
python3 pipeline/mnq_ml/experiments/m1_pullback_scalper/build_l1_context.py --dry-run
python3 pipeline/mnq_ml/experiments/m1_pullback_scalper/build_m1_events.py --dry-run
python3 pipeline/mnq_ml/experiments/m1_pullback_scalper/train_candidate.py
```

`gate_training_data.py` is the hard preflight. Training must refuse to run
unless the gate writes `training_allowed: true`.

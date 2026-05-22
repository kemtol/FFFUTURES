# M1 SuperTrend Pullback Scalper

Research-only workspace for a standalone M1 scalping model. This is not a
5m CONS filter and does not use 5m Super Structure signals.

## Intent

Build a higher-frequency positive-expectancy scalper:

- Timeframe: M1 only.
- Core idea: SuperTrend continuation pullback.
- Entry: pullback touches the SuperTrend zone, then closes back in trend.
- Skip: long entries that are already overbought, short entries that are
  already oversold.
- Exit: condition-based TP/SL, not a fixed always-on exchange order design.
- Goal: more trade opportunities than 5m CONS while keeping drawdown bounded.

## Isolation

- Do not edit live router files.
- Do not edit `model/SUPER_STRUCTURE/meta_v7/*`.
- Do not reuse 5m CONS labels.
- Do not wire this to Telegram/TopstepX until research passes.

## Files

```text
pipeline/super_structure_ml/experiments/m1_st_pullback_scalper/
├── README.md
├── config.json
├── data_contract.md
├── idea_spec.md
├── build_l1_context.py
├── build_m1_events.py
├── validate_data_integrity.py
├── audit_lookahead.py
├── training_feature_whitelist.json
└── gate_training_data.py
```

Outputs:

```text
data/Level_1_Features/super_structure_ml/m1_st_pullback_context.parquet
data/Level_2_Datamart/super_structure_ml/m1_st_pullback_scalper_events.parquet
model/SUPER_STRUCTURE/m1_st_pullback_scalper/
```

## First Flow

```bash
python3 pipeline/super_structure_ml/experiments/m1_st_pullback_scalper/build_l1_context.py --dry-run --start-date 2026-01-01 --end-date 2026-02-01
python3 pipeline/super_structure_ml/experiments/m1_st_pullback_scalper/build_l1_context.py --force
python3 pipeline/super_structure_ml/experiments/m1_st_pullback_scalper/build_m1_events.py --dry-run --start-date 2026-01-01 --end-date 2026-02-01
python3 pipeline/super_structure_ml/experiments/m1_st_pullback_scalper/build_m1_events.py --force
python3 pipeline/super_structure_ml/experiments/m1_st_pullback_scalper/gate_training_data.py
```

`gate_training_data.py` is the hard P0 preflight. It checks L1/L2 schema,
non-null constraints, continuity, signal-to-entry timing, outcome-window gaps,
feature whitelist safety, and a look-ahead audit. Training scripts must refuse
to run unless this gate passes and writes `training_allowed: true`.

After this gate passes, add training and hypertuning in P1/P2.

## P0 Timing Contract

The event datamart uses executable next-bar timing:

```text
signal_ts = M1 bar t close
features  = only bar t and earlier
entry_ts  = M1 bar t+1
entry     = open(t+1) +/- slippage
```

Model feature whitelists must use `signal_*` and signal-derived columns only.
`entry_*`, `entry_price`, `risk_pts`, `sl_price`, `tp_price`, and outcome
columns are for execution/audit, not model decisions.

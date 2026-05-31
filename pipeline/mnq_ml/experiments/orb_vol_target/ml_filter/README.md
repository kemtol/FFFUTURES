# MNQ ORB ML Filter

This is the first ML overlay for the rule-based MNQ ORB baseline.

It is not a fork of the strategy. The rule-based baseline remains the control:

```text
15m OR, long only, TP 2R or 15:00 NY, risk $500
```

The ML layer should answer one narrow question:

```text
Should a valid baseline trade be taken full size, reduced, or skipped?
```

It should not replace the opening range breakout rules until the deterministic
baseline has passed the Topstep-style gates.

## Structure

```text
pipeline/mnq_ml/experiments/orb_vol_target/
├── README.md                # strategy umbrella and rule-based baseline
├── sweep_orb_params.py      # deterministic parameter sweep
└── ml_filter/
    ├── config.json
    ├── build_ml_dataset.py
    └── README.md

data/Level_2_Datamart/mnq/orb_vol_target/
├── sweeps/                  # shared rule-based sweep artifacts
└── ml_filter/               # ML-ready datasets and manifests

model/MNQ/orb_vol_target/
└── ml_filter/               # future trained model artifacts
```

## Dataset Contract

Input:

```text
data/Level_2_Datamart/mnq/orb_vol_target/sweeps/sweep_events.parquet
```

Output:

```text
data/Level_2_Datamart/mnq/orb_vol_target/ml_filter/candidate_a_dataset.parquet
data/Level_2_Datamart/mnq/orb_vol_target/ml_filter/candidate_a_dataset_manifest.json
```

Current dataset filters the frozen Candidate A row family:

```text
orb_minutes = 15
side_mode = long
exit_mode = tp_2r_or_time
target_risk_usd = 500
```

Feature columns are intentionally limited to information known at signal/entry
time. Exit price, PnL, and R multiple are labels/evaluation fields, not model
features.

## Command

```bash
python3 pipeline/mnq_ml/experiments/orb_vol_target/ml_filter/build_ml_dataset.py --force
```

## Next Steps

1. Train a simple baseline classifier on the dataset.
2. Evaluate as a filter/sizing overlay against the unchanged rule-based
   baseline.
3. Rank by Topstep-style rolling 30D pass rate, MLL, and consistency, not just
   classification metrics.

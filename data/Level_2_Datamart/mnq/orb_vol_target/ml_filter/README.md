# MNQ ORB ML Filter Datamart

ML-ready datasets derived from the deterministic MNQ ORB rule-based baseline.

This folder must stay separate from `sweeps/` so the rule-based control remains
clean and reproducible.

Current expected artifacts:

| File | Purpose |
| --- | --- |
| `candidate_a_dataset.parquet` | Model-ready trade-level dataset for Candidate A |
| `candidate_a_dataset_manifest.json` | Row counts, split counts, feature list, and no-lookahead note |

Candidate A:

```text
15m OR, long only, TP 2R or 15:00 NY, risk $500
```

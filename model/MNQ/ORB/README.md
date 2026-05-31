# MNQ ORB

This is the MNQ ORB model namespace. Use this folder as the entry point when
asking: "model mana yang sedang dibahas, yang dipakai yang mana?"

## Current Decision Map

There is no promoted live MNQ model yet.

| Role | Model / Strategy ID | Status | Folder |
| --- | --- | --- | --- |
| Current rule-based candidate | `rule_based_15m_long_tp2r_eod` | Best current baseline, not live-ready | `model/MNQ/ORB/rule_based_15m_long_tp2r_eod/` |
| ML risk-sizing overlay | `orb_risk_adjusted_model` | Research only, not used for live sizing yet | `model/MNQ/ORB/orb_risk_adjusted_model/` |
| Old placeholder | `ml_filter` | Parked placeholder | `model/MNQ/ORB/ml_filter/` |

## What "Used" Means Right Now

Current working baseline:

```text
rule_based_15m_long_tp2r_eod
```

Primary report:

```text
model/MNQ/ORB/rule_based_15m_long_tp2r_eod/REPORT.md
```

Contract:

```text
MNQ, NY session, 15m opening range, long only,
first M1 close above OR high, entry next M1 open,
TP 2R or 15:00 NY time exit, risk $500 baseline,
no normal strategic SL.
```

This is the candidate to compare against. It is not yet a live deployment.

The ML folder exists to answer whether probabilities can improve risk sizing.
It is not the current trading engine and should not be treated as the live
candidate until promotion gates pass.

## Why Data Artifacts Still Live Under `data/`

`data/` is for reproducible lineage:

- raw bars
- feature tables
- datamarts
- audit reports
- sweep outputs
- canonical event parquet files

`model/` is for selected candidates:

- model cards
- strategy contracts
- trained model files
- reports and metrics
- deployment/promotion notes
- pointers back to the exact data artifacts used

So the clean workflow is:

```text
pipeline/ builds it
data/ stores the reproducible artifacts
model/ names the candidate and says whether it is usable
```

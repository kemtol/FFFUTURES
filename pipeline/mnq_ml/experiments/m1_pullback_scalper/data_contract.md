# MNQ M1 Data Contract

## Namespaces

MNQ research must use these paths:

```text
pipeline/mnq_ml/
data/Level_1_Features/mnq/
data/Level_2_Datamart/mnq/
model/MNQ/
```

It must not read or write these Gold/MGC paths:

```text
pipeline/super_structure_ml/
data/Level_1_Features/super_structure_ml/
data/Level_2_Datamart/super_structure_ml/
model/SUPER_STRUCTURE/
```

## L0 Source

Expected raw schema:

```text
timestamp_utc  timezone-aware UTC timestamp
open           float
high           float
low            float
close          float
volume         numeric
```

`config.json` starts with `source.ready=false`. Builders must refuse to produce
training data until the MNQ source is explicitly marked ready.

## L1 Context

L1 context is one row per M1 bar. Required base columns:

```text
timestamp_utc
open
high
low
close
volume
prev_gap_seconds
data_quality_ok
```

Indicator warmup nulls are allowed only before rows become eligible for events.

## L2 Events

L2 events use executable next-bar timing:

```text
signal_ts = closed M1 bar t
features  = bar t and earlier only
entry_ts  = M1 bar t+1
entry     = open(t+1) +/- slippage
```

Audit/execution columns are not model features:

```text
entry_*
entry_price
risk_pts
sl_price
tp_price
exit_*
hold_bars
label
pnl_usd
```

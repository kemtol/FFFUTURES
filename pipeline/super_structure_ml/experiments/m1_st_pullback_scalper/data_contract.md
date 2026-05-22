# Data Contract

This experiment uses an explicit L0 -> L1 -> L2 contract.

## L0 Raw

Source:

```text
data/Level_0_Raw/MGC_1m.db
```

Rules:

- Read-only.
- One row per M1 candle.
- Must satisfy OHLC invariants: `high >= open/close/low`, `low <= open/close/high`.

## L1 Context

Output:

```text
data/Level_1_Features/super_structure_ml/m1_st_pullback_context.parquet
```

L1 stores one row per raw M1 candle:

- Raw OHLCV: `open`, `high`, `low`, `close`, `volume`
- Causal indicators: SuperTrend, ATR, ADX, CCI, RSI, DEMA, CT VWAP
- Rolling features computed only from current/past bars

L1 also writes a manifest with row count, column list, SHA-256, date range,
source config, indicator config, and null rates.

## L2 Events

Output:

```text
data/Level_2_Datamart/super_structure_ml/m1_st_pullback_scalper_events.parquet
```

L2 must derive from L1. Every event row includes:

- Event metadata and label/outcome
- `signal_ts`: M1 candle close that generated the signal
- Signal OHLCV/indicator snapshot copied from L1 at `signal_ts`
- `entry_ts`: next M1 bar used for executable entry
- Entry OHLCV snapshot copied from L1 at `entry_ts`
- Risk/exit/cost fields

Execution timing:

```text
signal candle = bar t
features      = bar t and earlier only
entry         = open of bar t+1 plus/minus slippage
```

Training features may not include:

- future-outcome columns: `pnl_usd`, `label`, `exit_price`, `exit_reason`, `exit_ts`, `hold_bars`
- execution/pricing columns: `entry_*`, `entry_price`, `entry_gap_seconds`, `risk_pts`, `sl_price`, `tp_price`

The execution/pricing columns are stored for simulation audit, not model
decision features.

## Validation

Hard gate before training:

```bash
python3 pipeline/super_structure_ml/experiments/m1_st_pullback_scalper/gate_training_data.py
```

The gate writes:

```text
data/Level_2_Datamart/super_structure_ml/m1_st_pullback_scalper_training_gate.json
```

Training is allowed only when this report has `status: PASS` and
`training_allowed: true`.

Validator coverage:

- L1 sorted and unique timestamps
- L1 OHLCV invariants
- Required L1/L2 columns exist
- Every L2 `signal_ts` and `entry_ts` exists in L1
- L2 signal OHLCV exactly matches L1 source bar at `signal_ts`
- L2 entry OHLCV exactly matches L1 source bar at `entry_ts`
- `entry_ts > signal_ts`
- L2 feature columns have no nulls
- L1 hard columns have no nulls
- L2 event rows have no nulls
- Whitelisted model features have no nulls
- Signal-to-entry gap is exactly one M1 bar
- L2 signals do not come from quarantined data-quality windows
- Outcome windows do not cross M1 continuity gaps
- Whitelisted features do not include execution or future/outcome fields

## Look-Ahead Audit

Run:

```bash
python3 pipeline/super_structure_ml/experiments/m1_st_pullback_scalper/audit_lookahead.py --sample 500
```

The audit classifies columns as:

- model-safe direct snapshot from L1 at `signal_ts`
- model-safe derived values from L1 at `signal_ts`
- execution/pricing columns from L1 at `entry_ts`
- future outcome/label columns
- metadata/cost columns

Execution and future outcome columns are valid audit fields, but must never be
passed as model features.

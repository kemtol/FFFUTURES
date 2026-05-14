# Super Structure ML

Research workspace for ML/meta-modeling around the Super Structure strategy.

As of 2026-05-14, V8 router is live (Meta-v7 Refined CONS + v1.12 AGGR
mechanical). See `plans/super_structure_ml_v8.md` for current roadmap and
`pipeline/live/inference_router.py` for the production decision path.

## Current Status (V8 LIVE)

SMART_1 dual-mode is now wired through `pipeline/live/inference_router.py`:

| Mode | Purpose | Production path | Event source | Status |
| --- | --- | --- | --- | --- |
| Conservative | High-confidence ST-flip trades | `model/SUPER_STRUCTURE/meta_v7/inference_model.txt` + `inference_config_refined.json` (dynamic threshold per session_cluster) | DEMA-cross signal in `super_structure.py` main loop | LIVE (toggle: `USE_V8_ROUTER=True`) |
| Aggressive | DEMA/SuperTrend pullback scalper, RR 1:1 | **Mechanical** filter `risk_pts <= 12` on v1.12 events. No ML brain. | `pipeline/live/pullback_detector.py` emits events per 5m bar | LIVE |

Combined walk-forward (90d): PnL +$5,151, max DD -$1,861 → PASS Topstep
(borderline; $139 headroom). Sync-verified vs `simulate_cons_ml_aggr_mech.py`
at 0/0 divergence.

Legacy SMART_1 dual-ML (regime_dispatcher + cons_brain + aggr_brain) tetap
di-load di `_load_smart_models()` untuk fallback path saat `USE_V8_ROUTER=False`.
**Do not delete `aggressive_brain.txt`** — rollback safety.

Important: while V8 is live, treat any NEW model experiment as research until
walk-forward PASS + sync verify before promoting.

## Layout

- `sources/`: raw inputs and extraction helpers.
- `features/`: feature builders for entry-time and context features.
- `labels/`: label builders and target definitions.
- `datasets/`: reserved for versioned local dataset exports.
- `train/`: datamart builders and model training entry points.
- `eval/`: replay, Topstep-style audit, and validation scripts.
- `analysis/`: exploratory analysis, visualization, Monte Carlo, feature discovery.
- `reports/`: generated metrics and comparison summaries.
- `artifacts/`: model-ready exports, thresholds, config snapshots.

Most current parquet/model outputs are under:

- `data/Level_2_Datamart/super_structure_ml/`
- `model/SUPER_STRUCTURE/`

## Rules

1. Keep the live strategy untouched.
2. Add new research code with explicit versioning.
3. Never overwrite older datamarts/models during discovery.
4. Do not promote an aggressive model just because 2026 PnL looks good.
5. For Topstep relevance, evaluate pass/MLL survival, not AUC alone.
6. For aggressive pullback research, acknowledge this is a recent gold-regime thesis, not an all-history thesis.

## Datamart Registry

Status language:

- `ACTIVE`: current source of truth for the next research step.
- `ANCHOR`: accepted baseline for one mode, but not the focus of current aggressive work.
- `RETROSPECTIVE`: useful for postmortem/comparison only.
- `LEGACY`: older experiment; do not extend unless explicitly resurrected.
- `STERILE`: read-only artifact. Do not mutate or overwrite.

| Datamart | Strategy/mode | Status | Sterility | Notes |
| --- | --- | --- | --- | --- |
| `data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet` | Conservative ST-flip selector | `ANCHOR` | `STERILE` | Used by `meta_v7`; keep as conservative reference. |
| `data/Level_2_Datamart/super_structure_ml/v1_12_training_datamart.parquet` | Aggressive DEMA/ST pullback RR 1:1 | `ACTIVE` | Regeneratable by v1.12 builder, do not hand-edit | Current aggressive baseline for bucket/rule discovery. |
| `data/Level_2_Datamart/super_structure_ml/v1_11_training_datamart.parquet` | Aggressive pullback + DEMA family + macro | `RETROSPECTIVE` | `STERILE` | Produced `v1_11_deep`; keep for audit, do not use as baseline. |
| `data/Level_2_Datamart/super_structure_ml/v1_10_training_datamart.parquet` | Aggressive macro-enhanced experiment | `LEGACY` | `STERILE` | Macro experiment before v1.11; not current. |
| `data/Level_2_Datamart/super_structure_ml/v1_8_training_datamart.parquet` | Aggressive enriched pullback experiment | `LEGACY` | `STERILE` | Earlier DEMA/pullback lineage. |
| `data/Level_2_Datamart/super_structure_ml/pullback_events_enriched.parquet` | Aggressive DEMA100 pullback events | `LEGACY` | `STERILE` | Older event source used by some SMART_1 audit scripts. |
| `data/Level_2_Datamart/super_structure_ml/pullback_events.parquet` | Early pullback events | `LEGACY` | `STERILE` | RR 1.5 / older proximity logic; not aligned with current baseline. |
| `data/Level_2_Datamart/super_structure_ml/v6_advanced_features.parquet` | Conservative/meta feature experiment | `LEGACY` | `STERILE` | Pre/current conservative feature lineage. |
| `data/Level_2_Datamart/super_structure_ml/v5_raw_expanded.parquet` | Conservative/meta raw expansion | `LEGACY` | `STERILE` | Older feature expansion. |
| `data/Level_2_Datamart/super_structure_ml/latest_raw.parquet` | Historical scratch/latest export | `LEGACY` | `STERILE` | Do not assume semantic freshness from filename. |
| `data/Level_2_Datamart/super_structure_ml/v_production.parquet` | Older production-style export | `LEGACY` | `STERILE` | Do not use for aggressive v1.12 work. |

Rules for parquet handling:

- Do not patch parquet files manually.
- To change aggressive baseline mechanics, create `v1_13` builder and output.
- Keep v1.12 reproducible from `build_training_datamart_v1_12.py`.
- If an old script references `pullback_events_enriched.parquet` or `aggressive_brain.txt`, assume it is legacy until audited.
- Any file named `latest` is not automatically authoritative.

## Strategy Context

The live Super Structure mechanical strategy is trend-following:

- SuperTrend + DEMA + ADX + CCI.
- Live indicator constants in `pipeline/live/super_structure.py`:
  - `ST_FACTOR = 4.0`
  - `ATR_PERIOD = 12`
  - `DEMA_LENGTH = 200`
  - `ADX_LENGTH = 12`
  - `CCI_LENGTH = 12`
  - `CCI_SOURCE = "hl2"`
- Live entries require ADX/CCI/DEMA/ST alignment.
- Live exits use SL/trend-flip logic.

The aggressive research track is different. It is not the live ST-flip
strategy. It is a pullback scalper:

- Candidate direction follows SuperTrend regime.
- Entry waits for a pullback near the SuperTrend line.
- DEMA100 is used as directional filter.
- Candle color confirms rejection.
- SL is anchored around SuperTrend with a 1.0 point buffer.
- TP is RR 1:1.

## Retrospective: v1.11 Deep Failure

Latest aggressive deep model audited:

- Model: `model/SUPER_STRUCTURE/SMART_1/aggressive_brain_v1_11_deep.txt`
- Trainer: `pipeline/super_structure_ml/train/train_smart_1_aggressive_v1.11_deep.py`
- Datamart: `data/Level_2_Datamart/super_structure_ml/v1_11_training_datamart.parquet`

Observed audit:

| Metric | Value |
| --- | ---: |
| Train rows | 15,699 |
| OOT rows | 1,862 |
| Train AUC | 1.0000 |
| OOT AUC | ~0.5059 |
| OOT threshold 0.55 trades | 678 |
| OOT threshold 0.55 PnL | +$2,236 |
| OOT threshold 0.55 max DD | ~-$4,661 |

Diagnosis:

- Main issue was training method:
  - `num_boost_round=1000`
  - `num_leaves=63`
  - `min_data_in_leaf=20`
  - no early stopping
  - result: model memorized train noise.
- Datamart also contributed:
  - candidate base expectancy was weak/negative across broad history.
  - feature-label correlations were small.
  - macro features appeared strong in train but were regime-sensitive.
  - v1.11 indicator params did not match live defaults:
    - v1.11 used ST ATR 10, ADX 14, CCI 20.
    - live uses ST ATR 12, ADX 12, CCI 12.
  - macro fields had non-trivial missingness and caused row dropping.
- Probability scores were not reliable:
  - OOT correlation of `prob` to `label` was near zero.
  - OOT correlation of `prob` to `pnl_usd` was near zero.
  - top probability deciles were not consistently better.

Decision:

- Do not use `aggressive_brain_v1_11_deep.txt` as baseline.
- Do not wire it to inference.
- Use v1.12 datamart for the next aggressive discovery pass.

## Current Aggressive Baseline: v1.12

New builder:

- `pipeline/super_structure_ml/train/build_training_datamart_v1_12.py`

New output:

- `data/Level_2_Datamart/super_structure_ml/v1_12_training_datamart.parquet`

Command:

```bash
python3 pipeline/super_structure_ml/train/build_training_datamart_v1_12.py
```

Validation:

```bash
python3 -m py_compile pipeline/super_structure_ml/train/build_training_datamart_v1_12.py
```

### v1.12 Design

Purpose: create a clean mechanical baseline for aggressive pullback discovery
before ML.

Changes versus v1.11:

- Uses live-aligned indicator params:
  - ST factor 4.0
  - ATR period 12
  - ADX length 12
  - CCI length 12, source `hl2`
- Keeps DEMA pullback scalper, RR 1:1.
- Uses DEMA100 directional filter.
- Keeps DEMA50/100/200 distance and slope features.
- Removes macro inputs for baseline:
  - no `oil_return`
  - no `us10y_change`
  - no `dxy_return`
- Changes pullback proximity from percentage-of-price to ATR band:
  - `pullback_band = max(0.5 points, atr * 0.25)`
- Uses 100-bar max hold.
- Timeout outcome is mark-to-market at final close, not automatic loss.

Candidate rules:

| Side | Rule |
| --- | --- |
| Long | ST direction stays bullish (`-1`), close > DEMA100, close > ST, low touches `ST + band`, close > open |
| Short | ST direction stays bearish (`+1`), close < DEMA100, close < ST, high touches `ST - band`, close < open |

Risk/target:

- Long SL: `ST - 1.0`
- Short SL: `ST + 1.0`
- TP: 1R from entry
- Commission: $1.74 round-turn
- MGC point value: $10/point

### v1.12 Columns

Core audit/output:

- `pullback_id`
- `entry_ts`
- `side`
- `entry_price`
- `sl_price`
- `tp_price`
- `exit_ts`
- `exit_price`
- `exit_reason`
- `hold_bars`
- `risk_pts`
- `label`
- `pnl_pts`
- `pnl_usd`

Features:

- `dist_d50_atr`
- `dist_d100_atr`
- `dist_d200_atr`
- `d50_slope`
- `d100_slope`
- `d200_slope`
- `close_slope_5`
- `dema_stack`
- `entry_adx`
- `entry_cci`
- `cci_abs`
- `rsi_7`
- `wick_ratio`
- `candle_body_atr`
- `bar_range_atr`
- `st_gap_ratio`
- `touch_distance_atr`
- `pullback_band_atr`
- `hour_utc`
- `dow`
- `session_cluster`

### v1.12 Audit Snapshot

Generated artifact:

- Rows/events: 1,471
- Date range: 2023-01-03 21:00 UTC to 2026-05-06 13:55 UTC
- File size: ~355 KB
- No missing values in top audit fields.

All-history:

| Metric | Value |
| --- | ---: |
| Trades | 1,471 |
| Win rate | 42.56% |
| Avg PnL/trade | -$31.10 |
| Total PnL | -$45,755 |
| Max DD | -$48,309 |

2026 only:

| Metric | Value |
| --- | ---: |
| Trades | 76 |
| Win rate | 61.84% |
| Avg PnL/trade | +$24.40 |
| Total PnL | +$1,854 |
| Max DD | -$592 |

720d train / 200d OOT split anchored to latest timestamp:

| Split | Trades | Win rate | Avg PnL/trade | Total PnL | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 482 | 43.98% | -$39.07 | -$18,833 | -$19,509 |
| OOT | 126 | 61.11% | +$15.58 | +$1,963 | -$592 |

Exit breakdown:

| Exit | Trades | Avg PnL |
| --- | ---: | ---: |
| SL | 819 | -$81.36 |
| TP | 612 | +$38.44 |
| TIMEOUT | 40 | -$66.12 |

Interpretation:

- v1.12 is not an all-history edge.
- v1.12 has clear recent/OOT strength.
- This matches the working thesis: gold in 2024-2026 is a different volatility regime.
- Do not optimize for all-time if the target is Topstep under current high-volatility gold conditions.

## Research Direction

The next goal is not deep ML. The next goal is to isolate the recent-regime
mechanical edge.

Use v1.12 for bucket and rule discovery:

- side
- session/hour
- day of week
- ADX bucket
- CCI bucket
- RSI bucket
- `risk_pts`
- `st_gap_ratio`
- `touch_distance_atr`
- `bar_range_atr`
- `dema_stack`
- DEMA slopes
- hold duration / timeout behavior

Questions to answer before training:

1. Which 2026-positive buckets are robust?
2. Which 2024-2025-negative buckets should be avoided?
3. Is the edge mostly session-specific?
4. Is the edge mostly long or short?
5. Is 1R optimal, or should RR / SL buffer be tuned?
6. Are TIMEOUT trades structurally bad enough to need an earlier time stop?
7. Does risk size (`risk_pts`) explain the bad historical performance?

## Feature Fix Backlog

The next work should fix feature reliability before training another model.

### Must Fix Before ML

1. Add a bucket audit script for v1.12.
   - Risk quick-win implemented at `pipeline/super_structure_ml/analysis/audit_v1_12_risk.py`
   - Risk reports:
     - `model/SUPER_STRUCTURE/SMART_1/reports/v1_12_risk_bucket_audit.csv`
     - `model/SUPER_STRUCTURE/SMART_1/reports/v1_12_risk_cutoff_audit.csv`
   - Still needed general bucket audit path: `pipeline/super_structure_ml/analysis/audit_v1_12_buckets.py`
   - Input: `v1_12_training_datamart.parquet`
   - Output: console tables or CSV under `model/SUPER_STRUCTURE/SMART_1/reports/`
   - Required buckets:
     - year/month
     - side
     - session_cluster/hour_utc
     - day of week
     - `entry_adx`
     - `entry_cci` / `cci_abs`
     - `rsi_7`
     - `risk_pts`
     - `st_gap_ratio`
     - `touch_distance_atr`
     - `bar_range_atr`
     - `dema_stack`
     - DEMA slopes

   Risk quick-win finding:
   - Provisional cap: `risk_pts <= 12`.
   - Latest 30d aggressive v1.12 improves from +$211.74 / max DD -$592.14 to +$786.64 / max DD -$80.00.
   - 2026 aggressive v1.12 with this cap: 53 trades, +$1,543.60, max DD -$302.18.
   - OOT 200d with this cap: 98 trades, +$2,254.09, max DD -$302.18.
   - Treat this as a candidate mechanical filter, not final production logic; sample size is still small.

2. Validate v1.12 mechanical reproducibility.
   - Re-run `build_training_datamart_v1_12.py`.
   - Confirm row count and aggregate metrics do not unexpectedly drift.
   - If row count changes because source DB changed, document the new timestamp range and metrics.

3. Fix legacy script references.
   - Some SMART_1 eval scripts still load `pullback_events_enriched.parquet` and `aggressive_brain.txt`.
   - Do not use those scripts for v1.12 conclusions until they are versioned or copied to v1.12-specific scripts.
   - Preferred new scripts should explicitly say `v1_12` in filename.

4. Add Topstep-style evaluator for v1.12 mechanical rules.
   - Suggested path: `pipeline/super_structure_ml/eval/topstep_auditor_smart_1_v1_12.py`
   - Must report:
     - final PnL
     - max drawdown
     - MLL breach count
     - pass target hit
     - daily loss cap behavior
     - trades per day

5. Add train/OOT split helper.
   - Avoid each script inventing its own split.
   - Minimum supported modes:
     - fixed recent cutoff
     - rolling 12 months train / next N days OOT
     - 720d train / 200d OOT for continuity with previous audits

### Features To Revisit

These v1.12 features exist but need bucket validation before ML:

| Feature | Status | Why it matters |
| --- | --- | --- |
| `risk_pts` | Needs bucket audit | Large risk trades may explain bad historical drawdown. |
| `st_gap_ratio` | Needs bucket audit | Pullback too far from ST may not be a true pullback. |
| `touch_distance_atr` | Needs bucket audit | Measures how cleanly price touched ST band. |
| `entry_adx` | Needs bucket audit | Trend strength may separate 2026 edge from old regimes. |
| `entry_cci` / `cci_abs` | Needs bucket audit | Momentum exhaustion/rejection filter candidate. |
| `dema_stack` | Needs bucket audit | Trend alignment strength. |
| `d50_slope`, `d100_slope`, `d200_slope` | Needs bucket audit | Trend acceleration and regime quality. |
| `hour_utc`, `session_cluster` | Needs bucket audit | Edge may be session-specific. |
| `dow` | Needs bucket audit | Gold behavior may vary by weekday. |
| `exit_reason` / `hold_bars` | Needs audit | TIMEOUT losses may justify a time stop. |

### Features To Avoid For Now

Do not add these back until the non-macro baseline is understood:

- `oil_return`
- `us10y_change`
- `dxy_return`
- broad macro/regime joins
- any feature requiring forward-fill across missing market holidays

Reason: macro looked attractive in earlier experiments but made v1.11 easier to overfit and introduced missing-row behavior.

### Possible v1.13 Datamart Experiments

Only create these after v1.12 bucket audit:

- Sweep `PULLBACK_BAND_ATR`: 0.15, 0.20, 0.25, 0.30.
- Sweep `ST_BUFFER_PTS`: 0.5, 1.0, 1.5.
- Compare RR 0.75, 1.0, 1.25.
- Add explicit time stop before 100 bars if TIMEOUT remains bad.
- Add session gating directly into event generation only after bucket audit proves it.
- Add recent-regime tags, but keep them simple and causal:
  - rolling ATR percentile
  - rolling range percentile
  - rolling realized volatility

## Next Step Plan

Recommended immediate sequence:

1. Build a bucket audit script for v1.12.
   - Output tables for year, month, session, side, ADX bucket, CCI bucket, risk bucket, ST gap bucket.
   - Rank by OOT PnL, max DD, and sample count.
2. Create a mechanical rule baseline.
   - No ML yet.
   - Example: only recent-regime sessions/buckets that are positive and have enough samples.
3. Run Topstep-style simulation.
   - Start balance $50,000.
   - MLL $2,000.
   - Target +$3,000.
   - Include daily loss caps.
   - Report pass/fail/ruin, not just total PnL.
4. Only after a mechanical bucket baseline exists, train a small accept/reject model.
   - LightGBM `num_leaves` 7 or 15.
   - Early stopping required.
   - Use recent/rolling validation.
   - Select threshold by Topstep pass and MLL survival.
5. Add macro/regime features only after the non-macro baseline is stable.

## Training Guidance

Avoid:

- 1000 forced rounds without early stopping.
- Large trees on small event counts.
- Ranking models by AUC alone.
- Training on all history as if 2023 and 2026 are the same market.
- Adding macro features before the mechanical baseline is understood.

Prefer:

- Walk-forward or rolling validation.
- Recent-window training or recency weighting.
- Small LightGBM models.
- Simple mechanical filters as first baseline.
- Topstep pass/MLL objective.

Suggested initial LightGBM shape once bucket baseline exists:

```python
params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.02,
    "num_leaves": 7,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.7,
    "lambda_l1": 1.0,
    "lambda_l2": 2.0,
    "verbosity": -1,
}
```

Use early stopping. Then evaluate threshold by PnL/DD/Topstep objective.

## Promotion Checklist

Before any aggressive model can be considered for inference/live:

- Datamart lineage documented.
- Mechanical baseline reproduced.
- Walk-forward validation completed.
- 2026/recent OOT audited.
- Topstep pass/MLL simulation completed.
- Monte Carlo drawdown stress completed.
- Feature list frozen.
- Threshold frozen.
- Inference feature builder implemented separately.
- Inference output compared against offline datamart rows.
- No exchange execution changes made until reviewed.

## Known Artifacts

Conservative:

- `model/SUPER_STRUCTURE/meta_v7/inference_model.txt`
- `model/SUPER_STRUCTURE/meta_v7/inference_config.json`
- `model/SUPER_STRUCTURE/meta_v7/inference_config_refined.json`
- `data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet`

Aggressive older experiments:

- `model/SUPER_STRUCTURE/SMART_1/aggressive_brain.txt`
- `model/SUPER_STRUCTURE/SMART_1/aggressive_brain_rr1.txt`
- `model/SUPER_STRUCTURE/SMART_1/aggressive_brain_v1_8.txt`
- `model/SUPER_STRUCTURE/SMART_1/aggressive_brain_v1_10.txt`
- `model/SUPER_STRUCTURE/SMART_1/aggressive_brain_v1_11.txt`
- `model/SUPER_STRUCTURE/SMART_1/aggressive_brain_v1_11_deep.txt`

Aggressive current baseline:

- `pipeline/super_structure_ml/train/build_training_datamart_v1_12.py`
- `data/Level_2_Datamart/super_structure_ml/v1_12_training_datamart.parquet`

Daily memory:

- `_MEMORY/20260510.md`

## Final Working Thesis

The aggressive pullback scalper should be treated as a recent-regime gold
strategy. The objective is not to discover an all-time universal edge. The
objective is to identify whether the 2026-style high-volatility gold regime has
a repeatable pullback edge that can help SMART_1 pass Topstep 50K while staying
inside MLL constraints.

The next agent should continue from v1.12, not from v1.11 deep.

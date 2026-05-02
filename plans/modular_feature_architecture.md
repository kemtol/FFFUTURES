# Modular Feature Architecture — Futures Refactor Plan

> **Status**: ✅ **IMPLEMENTED** (2026-04-28) — The refactor is complete and 7 modules are in production.
> See [`research_program.md`](research_program.md) for the active research program and [`_MEMORY/20260428.md`](../_MEMORY/20260428.md) for cycle results.

## Current Problem

The current pipeline writes **everything** into a monolithic [`training_datamart_orb.parquet`](../data/Level_2_Datamart/training_datamart_orb.parquet):

- Core identifiers (date, session, orb_tf, breakout_ts, side)
- Labels (10 binary target columns)
- Context features (7 feature columns from [`build_market_context.py`](../pipeline/feature/build_market_context.py))
- Scale-invariant features (computed **inline in each sweep script**, duplicated across v2-v5)

To add a new feature (e.g., `breakout_candle_vol_ratio`), you must:
1. Modify [`build_market_context.py`](../pipeline/feature/build_market_context.py)
2. Rebuild the datamart from scratch
3. Or patch it — error-prone and slow

**This doesn't scale.** MMMACHINE/idx solved this with modular parquet files — let's copy that pattern.

---

## Grain Key

| Level | Grain | Example |
|-------|-------|---------|
| Breakout event (L1) | `(date, session, orb_tf, breakout_ts)` | `(2024-06-15, us, 15m, 2024-06-15 14:32:00)` |
| Datamart row (L2) | `(date, session, orb_tf, breakout_ts, side)` | Same + `rev` or `cont` |

**All features live at breakout-event grain** — same for rev and cont rows. When modules merge into the datamart, features duplicate across side rows, which is correct.

---

## Target Architecture

```
data/
├── Level_0_Raw/          (unchanged — SQLite DBs)
├── Level_1_Features/
│   ├── orb_ranges.parquet          (unchanged)
│   ├── breakout_events.parquet     (unchanged)
│   ├── market_context.parquet      (unchanged — backward compat)
│   ├── macro_data.parquet          ★ NEW (Cycle 4 — external macro data)
│   └── modules/                    ★ NEW
│       ├── orb_context_features.parquet
│       ├── scale_invariant_features.parquet
│       ├── volatility_normalized_features.parquet
│       ├── pre_breakout_profile_features.parquet
│       ├── session_momentum_features.parquet
│       ├── interaction_features.parquet
│       └── macro_features.parquet
└── Level_2_Datamart/
    └── training_datamart_orb.parquet  (core only: identifiers + labels)

pipeline/feature/
├── build_orb_ranges.py         (unchanged)
├── build_breakout_events.py    (unchanged)
├── build_market_context.py     (unchanged — still patches datamart for backward compat)
├── build_labels.py             (unchanged — still produces monolithic datamart)
└── modules/                    ★ NEW
    ├── __init__.py
    ├── loader.py
    ├── _TEMPLATE_generate_feature_module.py
    ├── generate_orb_context_features.py
    ├── generate_scale_invariant_features.py
    ├── generate_volatility_normalized_features.py
    ├── generate_pre_breakout_profile_features.py
    ├── generate_session_momentum_features.py
    ├── generate_interaction_features.py
    └── generate_macro_features.py

pipeline/fetch/
├── fetch_mgc_yfinance.py       (unchanged)
├── ingest_databento.py         (unchanged)
└── fetch_macro_data.py         ★ NEW (Cycle 4 — yfinance external data)

pipeline/analysis/
├── objective_sweep_orb_v4.py   (unchanged — legacy reproduction)
├── objective_sweep_orb_v5.py   (unchanged — legacy reproduction)
└── objective_sweep_orb_v6.py   ★ NEW — modular sweep script (7 modules active)
```

---

## Step-by-Step Implementation

### ✅ Step 1: Create `pipeline/feature/modules/loader.py`

A shared function that all modular sweep/training scripts call:

```python
EVENT_KEY = ["date", "session", "orb_tf", "breakout_ts"]

def load_features_from_modules(modules_dir: Path, core_df: pd.DataFrame) -> pd.DataFrame:
    """
    LEFT JOIN all *_features.parquet from modules_dir onto core_df.
    
    Each module must have grain (date, session, orb_tf, breakout_ts).
    Modules are merged in alphabetical file order. Warns on column conflicts.
    """
    module_files = sorted(modules_dir.glob("*_features.parquet"))
    if not module_files:
        print("[Modules] WARNING: no *_features.parquet found")
        return core_df
    
    df = core_df.copy()
    for fpath in module_files:
        module = pd.read_parquet(fpath)
        # Check column conflicts
        new_feats = set(module.columns) - set(EVENT_KEY)
        existing_feats = set(df.columns) - set(EVENT_KEY)
        overlap = new_feats & existing_feats
        if overlap:
            print(f"[Modules] ⚠️ COLUMN CONFLICT in {fpath.name}: {sorted(overlap)}")
        
        before = len(df)
        df = df.merge(module, on=EVENT_KEY, how="left")
        n_feat = len(module.columns) - len(EVENT_KEY)
        print(f"[Modules] {fpath.name}: {len(module):,} rows, {n_feat} feats, rows={before}→{len(df)}")
    return df
```

**Files created:**
- [`pipeline/feature/modules/__init__.py`](../pipeline/feature/modules/__init__.py) — empty
- [`pipeline/feature/modules/loader.py`](../pipeline/feature/modules/loader.py) — `load_features_from_modules()`

### ✅ Step 2: Create `pipeline/feature/modules/_TEMPLATE_generate_feature_module.py`

Adapted from the idx template. Key differences:

| Aspect | idx (stocks) | futures (ORB) |
|--------|-------------|---------------|
| Grain key | `(date, ticker)` | `(date, session, orb_tf, breakout_ts)` |
| Source data | `yfinance_1h.parquet`, `broksum_datamart.parquet` | `breakout_events.parquet`, SQLite DBs, `macro_data.parquet` |
| Output | `{family}_features.parquet` | `{family}_features.parquet` |
| Data load | `load_sources()` returns dict of DataFrames | Same pattern, different sources |

**Files created:**
- [`pipeline/feature/modules/_TEMPLATE_generate_feature_module.py`](../pipeline/feature/modules/_TEMPLATE_generate_feature_module.py)
  - `MODULE_NAME`, `GRAIN` config at top
  - `load_sources()` — breakout_events + SQLite DBs
  - `build_features(sources)` — implement feature logic
  - `check_column_conflict(df)` — check against existing modules
  - `main()` — argparse, dry-run, force, write

### ✅ Step 3: Create `generate_orb_context_features.py`

**Reads:** [`breakout_events.parquet`](../data/Level_1_Features/breakout_events.parquet) + 1m/15m SQLite DBs  
**Computes:** Same 7 features as [`build_market_context.py`](../pipeline/feature/build_market_context.py) lines 176-188:
- `orb_range_atr_ratio`, `day_of_week`, `time_in_session_min`
- `vwap_at_breakout`, `price_vs_vwap_pct`
- `adx_14_15m`, `ema_slope_1h`

**Output:** `data/Level_1_Features/modules/orb_context_features.parquet`  
**Grain:** `(date, session, orb_tf, breakout_ts)`

**Files created:**
- [`pipeline/feature/modules/generate_orb_context_features.py`](../pipeline/feature/modules/generate_orb_context_features.py)

### ✅ Step 4: Create `generate_scale_invariant_features.py`

**Reads:** [`breakout_events.parquet`](../data/Level_1_Features/breakout_events.parquet) (has `breakout_strength`, `atr14_at_entry`, `orb_range`)  
**Computes:** Same 7 features currently in [`add_scale_invariant_features()`](../pipeline/analysis/objective_sweep_orb_v5.py#L120-L130):
- `breakout_strength_atr_ratio`, `atr14_sq`, `breakout_strength_sq`
- `price_vs_vwap_pct_abs`, `orb_range_sq`, `adx_50_flag`, `breakout_strength_vs_orb`

**Output:** `data/Level_1_Features/modules/scale_invariant_features.parquet`  
**Grain:** `(date, session, orb_tf, breakout_ts)`

**Files created:**
- [`pipeline/feature/modules/generate_scale_invariant_features.py`](../pipeline/feature/modules/generate_scale_invariant_features.py)

### ✅ Step 5: Run module generators

```bash
cd /home-ssd/mkemalw/Projects/MMMACHINE/futures
python pipeline/feature/modules/generate_orb_context_features.py
python pipeline/feature/modules/generate_scale_invariant_features.py
```

Output files now exist in `data/Level_1_Features/modules/` with 7 modules.

### ✅ Step 6: Create `objective_sweep_orb_v6.py` (first modular sweep)

**Architecture:**
```
1. Load datamart → only CORE columns: identifiers + labels (no features)
2. Load feature modules via load_features_from_modules()
3. Run same Topstep sweep logic as v5
```

**Core columns definition:**
```python
CORE_COLS = [
    "date", "session", "orb_tf", "breakout_ts", "breakout_side", "side",
    "entry_price", "orb_range", "atr14_at_entry", "sl_dist", "breakout_strength",
    "session_close_ts",
    # All 10 labels
    "y_1r2_60m", "y_1r4_60m", "y_1r2_120m", "y_1r4_120m",
    "y_1r2_180m", "y_1r4_180m", "y_1r2_240m", "y_1r4_240m",
    "y_1r2_close60m", "y_1r4_close60m",
]
```

**Files created:**
- [`pipeline/analysis/objective_sweep_orb_v6.py`](../pipeline/analysis/objective_sweep_orb_v6.py)

### ✅ Step 7: Verify modular loading

```bash
python pipeline/analysis/objective_sweep_orb_v6.py
```

### ✅ Step 8: Update documentation

- [`README.md`](../README.md) — updated with modular architecture section and cycle results
- [`_MEMORY/20260427.md`](../_MEMORY/20260427.md) — v6 creation noted
- [`_MEMORY/20260428.md`](../_MEMORY/20260428.md) — cycles 1-4 logged
- [`plans/research_program.md`](research_program.md) — active research program

---

## Actual Results (2026-04-28)

| Cycle | Module | Features | AB Result | Decision |
|-------|--------|:--------:|:---------:|:--------:|
| 1 | `pre_breakout_profile_features` | 5 | All 10 improved | ✅ KEPT |
| 2 | `session_momentum_features` | 4 | 6/10 improved | ✅ KEPT |
| 3 | `interaction_features` | 5 | 4/10 improved | ✅ KEPT |
| 4 | `macro_features` | 4 | 4/10 improved (Δ +0.1928) | ✅ KEPT |
| **Total** | **7 modules** | **42** | **All 10 targets positive** | |

**Active modules directory:**
```
data/Level_1_Features/modules/
├── orb_context_features.parquet         # 7 features (V2 context)
├── scale_invariant_features.parquet     # 7 features (V3 scale-invariant)
├── volatility_normalized_features.parquet # 5 features (vol-normalized)
├── pre_breakout_profile_features.parquet  # 5 features (pre-breakout candles)
├── session_momentum_features.parquet      # 4 features (session momentum)
├── interaction_features.parquet           # 5 features (interaction terms)
└── macro_features.parquet                 # 4 features (SPX/DXY/US10Y/Oil)
```

## Key Learning: Date Dtype for Modules

All modules must store `date` as `datetime.date` objects (not str). The module `_TEMPLATE` and `loader.py` have been updated with this requirement. String dates cause the LEFT JOIN to produce NaN on all feature columns.

---

## Backward Compatibility

| Script | Status | Why |
|--------|--------|-----|
| `build_orb_ranges.py` | Unchanged | Still produces L1 ORB data |
| `build_breakout_events.py` | Unchanged | Still produces L1 breakout events |
| `build_market_context.py` | Unchanged | Still patches datamart for legacy scripts |
| `build_labels.py` | Unchanged | Still produces monolithic datamart |
| `objective_sweep_orb_v4.py` | Unchanged | Still loads full datamart |
| `objective_sweep_orb_v5.py` | Unchanged | Still loads full datamart + inline features |
| `generate_orb_context_features.py` | **Created** | Writes module parquet (doesn't touch existing files) |
| `generate_scale_invariant_features.py` | **Created** | Writes module parquet |
| `generate_volatility_normalized_features.py` | **Created** | Writes module parquet |
| `generate_pre_breakout_profile_features.py` | **Created** | Writes module parquet |
| `generate_session_momentum_features.py` | **Created** | Writes module parquet |
| `generate_interaction_features.py` | **Created** | Writes module parquet |
| `generate_macro_features.py` | **Created** | Writes module parquet |
| `loader.py` | **Created** | Shared module loader |
| `objective_sweep_orb_v6.py` | **Created** | Modular sweep script |

No existing file is modified. The monolithic datamart still works for v1-v5 reproduction.

---

## Migration Path

```
Before:                         After:
build_labels.py                 build_labels.py
  ↓                               ↓
datamart (core + features)      datamart (CORE only: identifiers + labels)
  ↓                               ↓
sweep script (adds more feats)  modules/*.parquet → load_features_from_modules()
  ↓                               ↓
train model                     sweep script → train model
```

The `generate_*` scripts replace the manual feature-building step that was embedded in sweep scripts. Data flow becomes:

```
build_orb_ranges.py → orb_ranges.parquet
build_breakout_events.py → breakout_events.parquet
build_labels.py → datamart (core + labels)

generate_orb_context_features.py → modules/orb_context_features.parquet
generate_scale_invariant_features.py → modules/scale_invariant_features.parquet
generate_volatility_normalized_features.py → modules/volatility_normalized_features.parquet
generate_pre_breakout_profile_features.py → modules/pre_breakout_profile_features.parquet
generate_session_momentum_features.py → modules/session_momentum_features.parquet
generate_interaction_features.py → modules/interaction_features.parquet
generate_macro_features.py → modules/macro_features.parquet

sweep script → datamart + modules → train model
```

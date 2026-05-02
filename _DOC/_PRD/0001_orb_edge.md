# PRD-0002: ORB Edge — MGC Micro Gold Futures

## North Star

> **Profit Factor > 1.3 dan Sharpe Ratio > 1.0 pada out-of-sample walk-forward**
> (diukur setelah transaction costs, bukan gross)

Model berhasil jika setup yang lolos filter menghasilkan:
- Expectancy > +0.30R per trade (net setelah costs)
- Win rate reversal filtered > 60% (vs baseline ~55%)
- Max drawdown streak < 8 consecutive losses

---

## Objective

Baseline sudah menjawab bahwa **reversal selalu lebih baik dari continuation** di semua session.  
Pertanyaan yang tersisa — dan yang bernilai secara trading — adalah:

> "Dari semua reversal setup yang terbentuk hari ini, **mana yang worth diambil?**  
> Kondisi pasar seperti apa (ADX, VWAP position, waktu dalam session, ORB range) yang secara historis meningkatkan probabilitas reversal berhasil hit TP 2R dalam 120 menit?"

Output: model yang **mem-filter** setup reversal berkualitas tinggi dari yang noise, bukan sekadar probability matrix statis.

---

## User Flow

1. ORB terbentuk (15 menit setelah session open)
2. Harga breakout dari area ORB
3. User membuka tool → model mengevaluasi kondisi saat itu:
   - ADX, VWAP position, waktu dalam session, ORB range ratio, EMA slope
4. Tool output:
   - **"Reversal setup ini: HIGH / MEDIUM / LOW confidence"**
   - Probability hit TP 2R dalam 120m
   - Historical expectancy untuk kondisi serupa
5. User memutuskan: masuk reversal, skip, atau wait

---

## Sessions (UTC)

| Session | Open  | Close | ORB selesai (15m) |
|---------|-------|-------|-------------------|
| Tokyo   | 00:00 | 03:00 | 00:15             |
| London  | 07:00 | 10:00 | 07:15             |
| US      | 13:30 | 16:30 | 13:45             |

---

## ORB Definition

| ORB Type | Sumber      | Candles        |
|----------|-------------|----------------|
| ORB-5m   | MGC_5m.db   | 3 candle pertama (15 menit) |
| ORB-15m  | MGC_15m.db  | 1 candle pertama (15 menit) |
| ORB-30m  | MGC_15m.db  | 2 candle pertama (30 menit) |

Entry: setelah ORB selesai terbentuk, candle pertama yang close di luar range.

---

## Trade Setup

- **Entry**: close breakout candle pertama setelah ORB
- **SL**: 1.5 × ATR(14) dari entry
- **TP**: 2R atau 4R dari entry (relative to SL)
- **Sides**: Continuation (searah breakout) dan Reversal (berlawanan)
- **Holding periods**: T+60m, T+120m, T+180m, T+240m, hingga 60m sebelum close session

### Transaction Cost Assumption

| Komponen | Nilai | Catatan |
|----------|-------|---------|
| Spread   | 0.2 tick = $0.20/contract | MGC tick = $0.10 |
| Commission | $0.50/side = $1.00 round trip | estimasi retail |
| Slippage | 0.1 tick = $0.10 | entry saat close candle, bukan realtime |
| **Total per trade** | ~$1.30/contract | deduct dari gross P&L |

Setiap analisis expectancy harus mencantumkan **gross** dan **net** (setelah costs).

---

## Key Findings (Baseline — 16 tahun data, ~34K events)

### Reversal Win Rate per Tahun

| Periode | Win Rate 120m | Expectancy Gross | Catatan |
|---------|--------------|-----------------|---------|
| 2017–2018 | 65–68% | ~+0.97R | **Anomali** — dominasi average keseluruhan |
| 2016, 2019 | 43–44% | +0.28–0.32R | Moderat |
| 2011–2015 | 35–40% | +0.06–0.19R | Marginal |
| **2020–2026** | **31–36%** | **-0.05 to +0.09R** | **Regime terkini — hampir flat** |

- **Continuation selalu negative expectancy**: -0.10R hingga -0.15R gross di semua tahun
- Gold cenderung **mean-revert** setelah ORB breakout, tapi edge tidak stabil
- **2017–2018 adalah anomali** yang menggelembungkan overall average — bukan representasi kondisi terkini

### ADX sebagai Regime Discriminator

| ADX Level | Win Rate | Expectancy | Interpretasi |
|-----------|---------|-----------|-------------|
| < 20 | 35.1% | +0.052R | Ranging — edge tipis |
| 20–30 | 36.3% | +0.089R | Normal |
| 30–50 | 37.7% | +0.131R | Moderate trend |
| **> 50** | **54.0%** | **+0.621R** | **Strong trend — best regime** |

ADX > 50 di 2020–2026: wr=38.4%, exp=+0.152R — **3× lebih baik dari average recent years**.

> ⚠️ Baseline belum adjusted untuk transaction costs. Net expectancy reversal recent years kemungkinan ~0.00R hingga +0.05R tanpa filtering.

---

## Known Gaps & Risks

### 1. Label Ambiguity — Same-Candle TP/SL Hit
Dalam `first_hit()`, jika satu 1m candle punya `high >= TP` **dan** `low <= SL` sekaligus (news spike, gap), kita tidak tahu urutan hit-nya. Saat ini diasumsikan TP kena duluan → **overestimate win rate**.

**Mitigasi**: quantify berapa % candles yang ambiguous, pertimbangkan label tersebut sebagai `NaN` atau worst-case `0`.

### 2. Regime Instability
16 tahun data mencakup regime yang berbeda:

| Periode | Karakter | Risiko |
|---------|---------|--------|
| 2010–2012 | Gold bull run ke $1900 | Mean revert mungkin lebih lemah |
| 2013–2018 | Bear/sideways, volatility rendah | Edge mungkin berbeda |
| 2020 | COVID spike, vol ekstrem | Outlier distorsi label |
| 2022–2026 | Bull run baru ke $3000+ | Regime paling relevan untuk forward |

**Wajib**: analisis per-tahun untuk konfirmasi edge konsisten sebelum training.

### 3. Multiple Testing Problem
Kombinasi yang ditest: 3 ORB_tf × 3 session × 2 sides × 5 horizons × 2 targets = **180 kombinasi**.  
Dengan 16 tahun data, ada risiko menemukan "edge" yang sebenarnya adalah statistical noise.

**Mitigasi**: 
- Fokus pada satu primary target (`y_1r2_120m`) untuk training
- Gunakan strict out-of-sample holdout (2023–2026, ~3 tahun, tidak disentuh saat feature engineering)
- Bonferroni correction jika melakukan grid search hyperparameter

### 4. Walk-Forward Validation Wajib
Simple 70/30 split tidak valid untuk time series karena model bisa belajar dari future regime.

**Requirement**:
- Training window: rolling 3 tahun
- Validation window: 6 bulan berikutnya
- Minimum 8 fold untuk coverage yang cukup
- **Holdout final**: 2024–2026 tidak disentuh sampai model final

### 5. Execution Realism
Entry diasumsikan di `close` breakout candle — padahal dalam praktik, kita baru tahu candle close setelah close terjadi, dan order harus masuk di candle berikutnya.

**Opsi**: label ulang dengan entry di `open` candle berikutnya setelah breakout (lebih realistis).

---

## Label Schema (Level 2 Datamart)

Setiap breakout event menghasilkan **2 rows** — satu untuk continuation (`side="cont"`), satu untuk reversal (`side="rev"`).

| Kolom | Definisi |
|-------|---------|
| `y_1r2_Xm` | TP 2R kena sebelum SL dalam X menit (1=hit, 0=stopped) |
| `y_1r4_Xm` | TP 4R kena sebelum SL dalam X menit |
| `side` | "cont"=continuation, "rev"=reversal |

Holding periods X: 60, 120, 180, 240, close60m (60m sebelum session close).

> ⚠️ Label saat ini menggunakan entry = close breakout candle. Perlu validasi apakah realistis atau perlu geser ke open candle berikutnya.

---

## Feature List

### Implemented ✅ (ada di training_datamart_orb.parquet)

| Feature | Deskripsi |
|---------|-----------|
| `orb_range` | ORB high - low |
| `orb_range_atr_ratio` | orb_range / ATR(14) — median ~4× |
| `breakout_strength` | jarak close dari level yang ditembus |
| `atr14_at_entry` | ATR(14) pada 1m saat breakout |
| `sl_dist` | 1.5 × atr14_at_entry |
| `price_vs_vwap_pct` | (entry - session_vwap) / vwap × 100 |
| `adx_14_15m` | ADX(14) pada 15m — median ~29, range 6-98 |
| `ema_slope_1h` | sign EMA(20) slope pada 1h (1=up, -1=down) |
| `session` | tokyo / london / us |
| `orb_tf` | 5m / 15m / 30m |
| `side` | cont / rev |
| `breakout_side` | 1=up, -1=down |
| `day_of_week` | 0=Mon … 4=Fri |
| `time_in_session_min` | menit sejak session open saat breakout |

### Planned (belum diimplementasi)

| Feature | Prioritas | Deskripsi |
|---------|-----------|-----------|
| `breakout_candle_vol_ratio` | Medium | volume breakout candle / MA20 volume |
| `orb_cumvol_ratio` | Medium | total volume dalam ORB / average |
| `price_vs_prev_close_pct` | Medium | % dari close hari sebelumnya |
| `prev_session_range` | Medium | range session sebelumnya |
| `ema_slope_4h` | Medium | arah EMA(20) pada 4h |
| `bb_width_15m` | Low | Bollinger Band width pada 15m |
| `realized_vol_30m` | Low | std log returns 30m sebelum breakout |
| `atr_14_5m` | Low | ATR(14) pada 5m saat breakout |

---

## Current Datamart State

**`data/Level_2_Datamart/training_datamart_orb.parquet`**

- **68,374 rows** × **29 cols**
- Coverage: 2010–2026, ~4,000 trading days
- 2 rows per breakout event (cont + rev)
- 10 label columns: `y_1r2/y_1r4` × `60m/120m/180m/240m/close60m`

---

## Output: Probability Matrix

Untuk setiap (session, orb_tf, side):

| Hold Until | TP 2R Hit% | TP 4R Hit% | Expectancy Gross (R) | Expectancy Net (R) | Sample Size |
|------------|-----------|-----------|---------------------|-------------------|-------------|
| T+60m      | -         | -         | -                   | -                 | N           |
| T+120m     | -         | -         | -                   | -                 | N           |
| T+180m     | -         | -         | -                   | -                 | N           |
| T+240m     | -         | -         | -                   | -                 | N           |
| Close-60m  | -         | -         | -                   | -                 | N           |

Expectancy = (Win% × 2R) - (Loss% × 1R) untuk 1:2, atau (Win% × 4R) - (Loss% × 1R) untuk 1:4.

---

## Model (Phase 2)

- **Algorithm**: LightGBM binary classification
- **Primary target**: `y_1r2_120m` (reversal side only)
- **Scope**: satu model global, `session` + `orb_tf` + context features sebagai input
- **Tujuan**: filter reversal setup dengan probability > threshold, di mana threshold dikalibrasi untuk profit factor > 1.3

### Sample Weighting — Time-Based Exponential Decay

Edge tidak stabil across regimes (2017–2018 anomali). Solusi: **time-based sample weighting** dengan half-life 2 tahun.

```python
λ = ln(2) / 2  # half-life = 2 tahun
weight = exp(-λ × (max_year - year))
```

| Tahun | Bobot relatif |
|-------|--------------|
| 2026 | 1.00 |
| 2024 | 0.50 |
| 2022 | 0.25 |
| 2020 | 0.13 |
| 2018 | 0.06 |
| 2016 | 0.03 |

2017–2018 otomatis diminimalkan tanpa hard cutoff. Semua 34K events tetap digunakan.

### Validation Strategy (Wajib)

```
2010 ──────────────── 2020 ──── 2023 ──── 2026
│                              │          │
│  Walk-forward training       │  Holdout │
│  rolling 2y train / 6m val   │  FINAL   │
│  min 6 folds                 │  (lock   │
│                              │  dari    │
│  sample_weight applied       │  awal,   │
│  di setiap fold              │  tidak   │
│                              │  disentuh│
└──────────────────────────────┘  s.d.    │
                                 model    │
                                 final)   └─
```

- **Out-of-sample holdout**: Jan 2024 – sekarang, dikunci dari awal
- **Threshold kalibrasi**: menggunakan 2022–2023 saja (bukan full data)
- **Metric final**: Sharpe, profit factor, max drawdown, win rate — semuanya pada holdout

### Pre-Training Checklist
- [x] Regime analysis per tahun — edge tidak stabil, 2017–2018 anomali
- [x] ADX sebagai regime discriminator terkuat — ADX>50 di recent years = +0.152R
- [x] Keputusan training: time-based exponential weighting, half-life 2 tahun
- [ ] Quantify ambiguous labels (same-candle TP+SL hit)
- [ ] Compute net expectancy baseline (gross − transaction costs)
- [ ] Lock holdout set Jan 2024+

---

## Data Pipeline

```
Level_0_Raw/
  MGC_1m.db, MGC_5m.db, MGC_15m.db
      ↓
pipeline/feature/
  build_orb_ranges.py       → Level_1_Features/orb_ranges.parquet
  build_breakout_events.py  → Level_1_Features/breakout_events.parquet
  build_market_context.py   → Level_1_Features/market_context.parquet
  build_labels.py           → Level_2_Datamart/training_datamart_orb.parquet
      ↓
model/ORB/
  lgbm_orb_1r2_120m.txt
```

---

## Milestones

| # | Task | Output | Status |
|---|------|--------|--------|
| 1 | Session + ORB computation | `Level_1_Features/orb_ranges.parquet` | ✅ Done |
| 2 | Breakout detection + SL calculation | `Level_1_Features/breakout_events.parquet` | ✅ Done |
| 3 | Market context features | `Level_1_Features/market_context.parquet` | ✅ Done |
| 4 | Label generation (TP/SL check) | `Level_2_Datamart/training_datamart_orb.parquet` | ✅ Done |
| 4a | Regime analysis per tahun | notebook/analysis | Pending |
| 4b | Quantify ambiguous labels | inline check | Pending |
| 4c | Net expectancy baseline (after costs) | inline check | Pending |
| 5 | Probability matrix (baseline stats) | `edges/orb_breakout/analysis/` | Pending |
| 6 | LGBM training (walk-forward) | `model/ORB/` | Pending |
| 7 | Model inference tool | CLI / notebook | Pending |

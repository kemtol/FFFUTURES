# PRD-0002: ORB Edge — MGC Micro Gold Futures

## Objective

Berikan conviction kepada trader sebelum entry dengan menjawab:

> "Setelah ORB terbentuk dan harga breakout, apakah **continuation** atau **reversal** yang lebih worth secara historis — dengan SL = 1.5 ATR dan target R:R 1:2 atau 1:4 — jika posisi dipegang hingga T menit?"

Output bukan signal buy/sell, melainkan **probability matrix + expectancy** per kondisi, sehingga trader punya data untuk validasi atau menolak convictionnya sebelum entry.

---

## User Flow

1. ORB terbentuk (15 menit setelah session open)
2. Harga mulai breakout dari area ORB
3. User membuka tool → melihat tabel:
   - Continuation win%, expectancy
   - Reversal win%, expectancy
   - Per holding period: T+60m, T+120m, T+180m, ... hingga 60m sebelum close
4. User memutuskan: ikut continuation, reversal, atau skip

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

---

## Label Schema (Level 2 Datamart)

Setiap breakout event menghasilkan **2 rows** — satu untuk continuation, satu untuk reversal.

| Kolom | Definisi |
|-------|---------|
| `y_1r2_Xm` | TP 2R kena sebelum SL dalam X menit (1=hit, 0=stopped) |
| `y_1r4_Xm` | TP 4R kena sebelum SL dalam X menit |
| `side` | 1=continuation, -1=reversal |

Holding periods X: 60, 120, 180, 240, hingga session_close - 60m.

---

## Feature List (OHLCV-only, Phase 1)

### ORB Context
| Feature | Deskripsi |
|---------|-----------|
| `orb_range` | ORB high - low |
| `orb_range_atr_ratio` | orb_range / ATR(14) |
| `breakout_strength` | jarak close dari level yang ditembus |
| `breakout_candle_vol_ratio` | volume breakout candle / MA20 volume |
| `orb_cumvol_ratio` | total volume dalam ORB / average |

### Price Structure
| Feature | Deskripsi |
|---------|-----------|
| `price_vs_vwap` | posisi entry vs VWAP session |
| `price_vs_prev_close_pct` | % dari close hari sebelumnya |
| `price_vs_prev_session_high` | di atas/bawah high session sebelumnya |
| `price_vs_prev_session_low` | di atas/bawah low session sebelumnya |

### Volatility
| Feature | Deskripsi |
|---------|-----------|
| `atr_14_5m` | ATR(14) pada 5m saat breakout |
| `atr_14_15m` | ATR(14) pada 15m saat breakout |
| `bb_width_15m` | Bollinger Band width pada 15m |
| `realized_vol_30m` | realized volatility 30m sebelum breakout |

### Market Structure
| Feature | Deskripsi |
|---------|-----------|
| `ema_slope_1h` | arah EMA(20) pada 1h (1=up, -1=down) |
| `ema_slope_4h` | arah EMA(20) pada 4h |
| `prev_session_range` | range session sebelumnya |
| `htf_trend_aligned` | apakah breakout searah dengan trend 4h |

### Time Context
| Feature | Deskripsi |
|---------|-----------|
| `session` | tokyo=0, london=1, us=2 |
| `orb_tf` | 5, 15, atau 30 |
| `side` | 1=continuation, -1=reversal |
| `day_of_week` | 0=Mon … 4=Fri |
| `time_in_session_min` | menit sejak session open saat breakout |

---

## Output: Probability Matrix

Untuk setiap (session, orb_tf, side):

| Hold Until | TP 2R Hit% | TP 4R Hit% | Expectancy (R) | Sample Size |
|------------|-----------|-----------|---------------|-------------|
| T+60m      | -         | -         | -             | N           |
| T+120m     | -         | -         | -             | N           |
| T+180m     | -         | -         | -             | N           |
| T+240m     | -         | -         | -             | N           |
| Close-60m  | -         | -         | -             | N           |

Expectancy = (Win% × 2R) - (Loss% × 1R) untuk 1:2, atau (Win% × 4R) - (Loss% × 1R) untuk 1:4.

---

## Model (Phase 2)

Setelah baseline probability matrix tersedia:

- **Algorithm**: LightGBM binary classification
- **Target**: `y_1r2_120m` sebagai primary target (bisa dikonfigurasi)
- **Tujuan**: filter breakout mana yang lebih tinggi dari baseline expectancy
- **One model per (side, horizon)**: `model_cont_1r2_120m`, `model_rev_1r2_120m`, dst

---

## Data Pipeline

```
Level_0_Raw/
  MGC_1m.db, MGC_5m.db, MGC_15m.db
      ↓
Level_1_Features/
  orb_ranges.parquet        — ORB high/low per (date, session, orb_tf)
  breakout_events.parquet   — deteksi breakout + entry price + SL
  market_context.parquet    — ATR, VWAP, EMA slope, BB per candle
      ↓
Level_2_Datamart/
  training_datamart_orb.parquet  — 1 row per (breakout × side), features + labels
      ↓
model/ORB/
  lgbm_cont_1r2_120m.txt, dll
```

---

## Milestones

| # | Task | Output |
|---|------|--------|
| 1 | Session + ORB computation | `Level_1_Features/orb_ranges.parquet` |
| 2 | Breakout detection + SL calculation | `Level_1_Features/breakout_events.parquet` |
| 3 | Market context features | `Level_1_Features/market_context.parquet` |
| 4 | Label generation (TP/SL check) | `Level_2_Datamart/training_datamart_orb.parquet` |
| 5 | Probability matrix (baseline stats) | `edges/orb_breakout/analysis/` |
| 6 | LGBM training | `model/ORB/` |

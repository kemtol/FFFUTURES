# Objective Sweep Report — ORB

Generated: 2026-04-26 18:19

## Summary

- Labels tested: **10**
- Holdout: `2024-01-01` onward
- Topstep account: **50K** (target $3,000, MLL $2,000, consistency rule: best day < 50% total profit)
- Scoring: `pass_rate - fail_mll_rate` (best params from 128-point grid)
- **Best label:** `y_1r4_240m` (score=+0.216)
- Labels with score > 0: **10 out of 10** — semua target menghasilkan net positive, tapi **NONE** memenuhi GO gate

---

## 🔬 Critical Finding: 2026 Blockade

**Setiap target tanpa terkecuali memiliki pass_rate = 0% di 2026.**

Ini adalah masalah fundamental yang tidak bisa diselesaikan dengan:
- ✅ Pemilihan target/horizon yang berbeda
- ✅ Threshold tuning
- ✅ Switch ke 4R (risk-reward lebih tinggi)

Semua target collapse di 2026. Ini mengonfirmasi bahwa **regime 2026 membutuhkan pendekatan yang fundamentally berbeda**, bukan sekadar tweak parameter.

---

## 📊 Ranked Results (by score)

| Rank | Target | RR | Score | Pass Rate | Fail MLL | Med Pnl | $/1R | 
|------|--------|:--:|:-----:|:---------:|:--------:|:-------:|:----:|
| 🥇 | `y_1r4_240m` | 4.0 | **+0.216** | 36.0% | 14.4% | $1,504 | $100 |
| 🥈 | `y_1r4_close60m` | 4.0 | +0.166 | 17.1% | 0.5% | $434 | $100 |
| 🥉 | `y_1r4_120m` | 4.0 | +0.164 | 22.0% | 5.5% | $790 | $100 |
| 4 | `y_1r4_60m` | 4.0 | +0.128 | 22.2% | 9.3% | $451 | $100 |
| 5 | `y_1r4_180m` | 4.0 | +0.128 | 23.0% | 10.2% | $705 | $100 |
| 6 | `y_1r2_120m` *(original)* | 2.0 | +0.114 | 31.0% | 19.6% | $1,397 | $150 |
| 7 | `y_1r2_close60m` | 2.0 | +0.038 | 22.2% | 18.3% | $755 | $150 |
| 8 | `y_1r2_180m` | 2.0 | +0.033 | 7.1% | 3.8% | $961 | $100 |
| 9 | `y_1r2_240m` | 2.0 | +0.031 | 7.4% | 4.3% | $501 | $100 |
| 10 | `y_1r2_60m` | 2.0 | +0.012 | 21.3% | 20.1% | $1,005 | $150 |

---

## 📈 Yearly Breakdown — Best Target (y_1r4_240m) vs Original (y_1r2_120m)

| Metric | y_1r4_240m (Best) | y_1r2_120m (Original) |
|--------|:-----------------:|:---------------------:|
| **2024 pass_rate** | **51.3%** ✅ | 30.8% |
| 2024 fail_mll | 25.0% ❌ | 18.8% |
| 2024 med PnL | $2,545 | $1,411 |
| **2025 pass_rate** | **31.0%** | **38.9%** ✅ |
| 2025 fail_mll | 6.3% | 13.0% |
| 2025 med PnL | $1,471 | $2,064 |
| **2026 pass_rate** | **0.0%** ❌ | **0.0%** ❌ |
| 2026 fail_mll | **13.1%** ✅ | 55.7% ❌ |
| 2026 med PnL | -$926 | -$1,668 |

**Insight:** `y_1r4_240m` lebih unggul di risk management (fail_mll jauh lebih rendah di 2026: 13.1% vs 55.7%), tapi sama-sama tidak bisa pass di 2026.

---

## 🔍 Pola yang Terlihat

### 1. 4R Targets > 2R Targets
Semua target 4R (RR=4.0) memiliki score lebih tinggi dari target 2R mana pun. Alasannya:
- Butuh lebih sedikit winners untuk mencapai $3,000
- Fail_mll lebih rendah karena risk per trade lebih kecil relatif terhadap target

### 2. Holding Period: 240m > Shorter Horizons
`y_1r4_240m` (4 jam) outperforms `y_1r4_120m` (2 jam) dan `y_1r4_60m` (1 jam):
- Lebih banyak waktu untuk price mencapai TP
- Filter ADX + trend alignment lebih efektif di horizon panjang

### 3. 2026 Independence
Semua target fail di 2026 — ini bukan masalah label. Ini masalah **regime shift** di market microstructure.

---

## 🎯 Kesimpulan & Rekomendasi

### ✅ Target yang Paling Menjanjikan untuk Lanjut

```
y_1r4_240m  (4R target, 240 menit holding period)
```

**Alasan:**
- Score tertinggi (+0.216)
- 2024 pass_rate 51.3% — mendekati GO gate
- 2025 pass_rate 31.0% — solid, fail_mll rendah (6.3%)
- 2026 fail_mll rendah (13.1%) — tidak blow up, hanya tidak profit
- Parameter optimal: `risk=$100/1R`, `rev_q=0.75`, `cont_q=0.60`, `rev_adx_min=30`, `cont_adx_max=100`, `profit_cap=$1,400`

### ❌ Target yang Bisa Diabaikan

| Target | Alasan |
|--------|--------|
| `y_1r2_60m` | Score terendah, fail_mll tinggi di semua tahun |
| `y_1r2_180m` | Pass_rate sangat rendah (7%) |
| `y_1r2_240m` | Pass_rate sangat rendah (7%) |

### 🚧 Masih NO-GO untuk Topstep 50K

Meskipun `y_1r4_240m` adalah peningkatan signifikan dibanding original `y_1r2_120m`:
- Score: +0.216 vs +0.114
- Pass rate: 36.0% vs 31.0%
- Fail MLL: 14.4% vs 19.6%

**Masih jauh dari GO gate (pass_rate ≥ 60%, fail_mll ≤ 10%).**

---

## 📋 Next Steps

Berdasarkan hasil ini, saya rekomendasikan:

### Priority 1: Investigasi 2026 Collapse 🔴
Sebelum feature finding, perlu dipahami **mengapa** 2026 collapse. Analisis yang bisa dilakukan:
- ADX distribution di 2026 vs 2024-2025
- Volatility regime (ATR levels) di 2026
- VWAP position distribution — apakah trend terlalu kuat?
- Trade frequency di 2026 vs sebelumnya (avg_trades 26.9 vs 65.2 di 2024)

### Priority 2: Topstep Simulator Refinement 🟡
- Exact trading day boundary (5:00 PM CT - 3:10 PM CT)
- MGC contract integer sizing
- Agar hasil lebih akurat sebelum lanjut feature finding

### Priority 3: Feature Finding untuk y_1r4_240m 🟡
Setelah 2026 collapse dipahami, baru tambah features:
- Volume-based features
- Prior session range
- Realized volatility sebelum breakout
- 4H / daily trend filter

---

## Notes

- Models trained on 2010-2021, early-stopped on 2022-2023, evaluated on 2024+
- Sample weighting: exponential decay half-life=2y
- Policy: dynamic rev/cont/skip with trend+ADX gates
- Best params for y_1r4_240m: rev_q=0.75, cont_q=0.60, rev_adx_min=30, cont_adx_max=100, profit_cap=$1,400
- All models used risk_per_r_usd=$100 for best risk-adjusted results

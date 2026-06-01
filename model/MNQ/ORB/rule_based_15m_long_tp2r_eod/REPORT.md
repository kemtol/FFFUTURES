# Strategi MNQ Opening Range Breakout Rule-Based Iterasi v1
**Evaluasi Baseline 15m Long TP2R/EOD**

Tanggal laporan: **2026-06-01**

Model / strategy ID: `rule_based_15m_long_tp2r_eod`

Objective: **Topstep 50K research baseline** - mencari apakah breakout MNQ
setelah 15 menit pertama New York open punya positive expectancy yang cukup
untuk menjadi kandidat forward test.

Audience: trader futures, evaluator internal strategi MNQ, dan pembanding untuk
overlay machine learning.

---

## 1. Ringkasan Eksekutif

Laporan ini mengevaluasi strategi MNQ ORB v1 dari sudut pandang **baseline
rule-based**, bukan machine learning.

Aturan yang diuji sederhana: ambil posisi long setelah candle M1 pertama close
di atas high opening range 15 menit, entry pada open M1 berikutnya, lalu exit
di TP 2R atau time exit 15:00 New York. Strategi ini tidak memakai normal stop
loss; OR low hanya menjadi referensi sizing.

| Area | Hasil |
| --- | ---: |
| Periode sinyal | 2019-05-06 - 2026-05-26 |
| Total trade | 1,296 |
| Win rate | 56.48% |
| Net PnL | $33,091 |
| Max drawdown | -$12,124 |
| Profit factor | 1.12 |
| Daily Sharpe | 0.50 |
| Daily Sortino | 0.64 |
| 30D terakhir | 18 trade, $3,460 PnL, -$551 max DD |

**Kesimpulan utama:** baseline ini layak dipertahankan sebagai control strategy
karena window 30 hari terakhir menarik untuk objektif Topstep. Namun edge
historis panjangnya masih tipis: PF 1.12 dan Sharpe 0.50. Strategi belum
layak live tanpa simulasi MLL, consistency, catastrophic guard, dan forward
test.

---

## 2. Latar Belakang Strategi

Opening Range Breakout berangkat dari hipotesis bahwa rentang harga pada awal
sesi New York menyimpan informasi tentang imbalance intraday. Untuk Nasdaq
futures, tekanan order setelah cash open sering menjadi penentu arah sesi.

Versi ini sengaja dibuat sederhana:

1. Tidak memakai indikator tambahan.
2. Tidak memakai filter ML.
3. Tidak melakukan short continuation.
4. Tidak melakukan reversal.
5. Tidak memakai normal SL sebagai exit strategi.

Tujuannya adalah mendapatkan **baseline bersih**. Jika baseline saja tidak
punya edge, ML overlay akan mudah menjadi curve fitting. Jika baseline punya
edge, ML dapat diuji sebagai risk adjuster, bukan sebagai alasan untuk memaksa
trade.

---

## 3. Konteks Strategi

| Field | Value |
| --- | --- |
| Instrument | MNQ |
| Session | New York regular session |
| Source grain | Right-labeled M1 bars |
| Opening range | 15 minutes after 09:30 NY |
| Direction | Long only |
| Signal | First M1 close above OR high |
| Entry | Next M1 open after signal close |
| Exit | TP 2R first, otherwise 15:00 NY time exit |
| Normal strategic SL | None |
| Stop reference | OR low for position sizing only |
| Target risk | $500 |
| Max trades | 1 per NY session |

Model ini disebut rule-based karena semua keputusan entry dan exit ditentukan
oleh aturan eksplisit. Belum ada model probabilitas yang ikut menentukan trade.

---

## 4. Definisi Sizing

Baseline memakai target risk dollar tetap:

```text
contracts_float = target_risk_usd / risk_per_contract_usd
contracts_used = floor(contracts_float), minimum 1 contract
```

Dengan `target_risk_usd = $500`, jumlah kontrak otomatis turun saat OR/risk
melebar dan naik saat risk menyempit. Karena kontrak harus integer, actual risk
tidak selalu tepat $500.

Catatan penting: OR low **bukan** normal stop loss strategi. OR low hanya
referensi sizing. Exit tetap TP 2R atau time exit.

---

## 5. Hasil Historis 2019-2026

### 5.1 Equity Curve

![Equity Curve](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/equity_curve.png)

Equity curve menunjukkan strategi menghasilkan PnL positif secara historis,
tetapi jalurnya tidak linear. Ada fase panjang yang relatif datar dan beberapa
periode drawdown besar.

### 5.2 Drawdown

![Drawdown Curve](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/drawdown_curve.png)

Drawdown maksimum historis sebesar -$12,124. Ini jauh lebih
besar daripada batas MLL Topstep 50K, sehingga evaluasi live tidak boleh hanya
mengandalkan total PnL historis.

### 5.3 Monthly PnL

![Monthly PnL](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/monthly_pnl.png)

Grafik bulanan membantu melihat bahwa strategi tidak menghasilkan distribusi
profit yang stabil setiap bulan. Ada bulan kuat, bulan kosong, dan bulan rugi.

### 5.4 Distribusi PnL Per Trade

![Trade PnL Distribution](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/trade_pnl_distribution.png)

Rata-rata loss per trade masih lebih besar daripada rata-rata win. Edge muncul
dari kombinasi win rate 56.48%, sizing, dan beberapa periode momentum yang
produktif.

---

## 6. Metrik Historis

| Metric | Value |
| --- | ---: |
| Signal range | 2019-05-06 to 2026-05-26 |
| Trades | 1,296 |
| Win rate | 56.48% |
| Net PnL | $33,091 |
| Gross profit | $314,667 |
| Gross loss | -$281,576 |
| Profit factor | 1.12 |
| Max drawdown | -$12,124 |
| Return / DD | 2.73 |
| Expectancy / trade | $26 |
| Median trade | $93 |
| Average win | $430 |
| Average loss | -$499 |
| Payoff ratio | 0.86 |
| Average contracts | 3.48 |
| Max consecutive wins | 10 |
| Max consecutive losses | 6 |

---

## 7. Cost Model

| Cost | Value |
| --- | ---: |
| Commission + fees | $1.24 RT / contract |
| Slippage | 1 tick per side |
| Modeled slippage | $1.00 RT / contract |
| Total commission paid | $5,590 |
| Total modeled slippage | $4,508 |

Biaya sudah dimasukkan pada `pnl_net_usd`: TopstepX MNQ $1.24 round-turn per
contract dan modeled slippage 1 tick per side.

---

## 8. Daily Quality

Sharpe and Sortino are computed from daily dollar PnL over MNQ NY session days,
with zero PnL on no-trade days, annualized by `sqrt(252)`.

| Metric | Value |
| --- | ---: |
| Trading days measured | 2,197 |
| Active days | 1,296 |
| Active-day rate | 58.99% |
| Active-day win rate | 56.48% |
| Daily average PnL | $15 |
| Daily PnL std dev | $479 |
| Daily Sharpe | 0.50 |
| Daily Sortino | 0.64 |
| Best day | $991 |
| Worst day | -$3,781 |
| Best-day profit share | 2.99% |
| 50% consistency flag | Pass |

---

## 9. Rolling Window Terakhir

![Rolling Windows](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/rolling_windows.png)

| Window | Trades | Win Rate | PnL | Max DD |
| ---: | ---: | ---: | ---: | ---: |
| 5D | 1 | 100.00% | $101 | $0 |
| 10D | 5 | 60.00% | $895 | -$222 |
| 20D | 10 | 70.00% | $2,348 | -$222 |
| 30D | 18 | 72.22% | $3,460 | -$551 |
| 50D | 30 | 66.67% | $5,385 | -$861 |
| 100D | 54 | 55.56% | $4,035 | -$4,085 |
| 200D | 94 | 61.70% | $5,385 | -$4,561 |

Interpretasi:

- 30D terakhir adalah bagian paling menarik: 18 trade dan $3,460 PnL.
- 5D dan 10D masih terlalu pendek untuk menjadi bukti edge.
- 100D dan 200D tetap positif, tetapi DD historisnya mulai berat untuk Topstep.

---

## 10. SuperTrend Regime Filter Audit

SuperTrend audit ditambahkan untuk menjawab apakah drawdown March 2026 bisa
dikurangi dengan regime filter sederhana, tanpa langsung mengganti baseline.
Semua fitur dihitung dari bar yang sudah close dan di-join ke trade event
dengan rule `feature_ts <= signal_ts`.

### 10.1 Data Integrity

| Check | Value |
| --- | ---: |
| Feature family | `ST5_5`, `ST5_10`, `ST5_20`, `ST5_50`, `ST15_5`, `ST15_10`, `ST15_20`, `ST15_50` |
| SuperTrend factor | 4.00 |
| Direction convention | `-1 = bullish/up`, `+1 = bearish/down` |
| Join rule | Latest completed feature timestamp `<= signal_ts` |
| Lookahead violations | 0 |
| Max feature lag | 14 menit |


### 10.2 Perbandingan Variant Utama

#### Equity Curve

![ST5_50 Variant Equity Curve](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/supertrend_variant_equity_curve.png)

#### Drawdown Curve

![ST5_50 Variant Drawdown Curve](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/supertrend_variant_drawdown_curve.png)

#### Monthly PnL 2026

![ST5_50 Variant Monthly PnL 2026](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/supertrend_variant_monthly_pnl_2026.png)

#### Rolling Window PnL/DD

![ST5_50 Variant Rolling Windows](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/supertrend_variant_rolling_windows.png)

#### Trade PnL Distribution

![ST5_50 Variant Trade PnL Distribution](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/supertrend_variant_trade_pnl_distribution.png)

#### March 2026 Equity

![ST5_50 Variant March 2026 Equity](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/supertrend_variant_march_2026_equity.png)

| Variant | Trades | Long | Short | WR | PnL | DD | Ret/DD | Jan-May Trades | Jan-May PnL | Jan-May DD | Mar PnL | Mar DD | 30D Trades | 30D PnL | 30D DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Long only, no ST | 1,296 | 1,296 | 0 | 56.48% | $33,091 | -$12,124 | 2.73 | 72 | $6,096 | -$4,085 | -$2,633 | -$4,085 | 18 | $3,460 | -$551 |
| Long only, ST5_50 bullish | 964 | 964 | 0 | 56.85% | $28,199 | -$8,027 | 3.51 | 50 | $5,843 | -$1,916 | $102 | -$1,588 | 15 | $1,918 | -$551 |
| Long+Short, no ST | 1,767 | 905 | 862 | 53.03% | $26,501 | -$15,294 | 1.73 | 97 | $4,072 | -$3,623 | -$1,451 | -$3,105 | 21 | $839 | -$1,636 |
| Long+Short, ST5_50 aligned | 1,251 | 667 | 584 | 53.88% | $36,800 | -$9,099 | 4.04 | 63 | $7,328 | -$4,493 | $2,336 | -$1,029 | 16 | -$1,059 | -$1,894 |

Interpretasi:

- `Long only, ST5_50 bullish` adalah kandidat P0 paling bersih: hanya menambah
  satu rule regime filter, March 2026 membaik, dan sample size masih besar.
- `Long+Short, no ST` menambah frekuensi, tetapi short leg mentahnya tidak
  cukup kuat karena PnL full-history turun dan DD membesar.
- `Long+Short, ST5_50 aligned` menarik secara full-history dan March, tetapi
  30D terakhir negatif. Ini belum layak jadi kandidat utama tanpa investigasi
  stabilitas recent window.


### 10.3 Kandidat Kombinasi SuperTrend

Tabel ini menampilkan kandidat terbaik berdasarkan full-history return/DD,
dengan minimum `full_trades >= 100` dan `jan_may_2026_trades >= 30`.

| Candidate | N | Full Trades | Full PnL | Full DD | Ret/DD | Jan-May Trades | Jan-May PnL | Mar PnL | Mar DD | 30D PnL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ST5_5 & ST5_20 & ST5_50 & ST15_20 | 4 | 496 | $30,466 | -$5,351 | 5.69 | 32 | $5,403 | $229 | -$1,461 | $1,431 |
| ST5_5 & ST5_20 & ST15_20 | 3 | 513 | $30,428 | -$5,351 | 5.69 | 33 | $5,278 | $103 | -$1,587 | $1,431 |
| ST5_5 & ST5_50 & ST15_20 | 3 | 500 | $30,226 | -$5,351 | 5.65 | 32 | $5,403 | $229 | -$1,461 | $1,431 |
| ST5_5 & ST5_10 & ST5_20 & ST5_50 & ST15_20 | 5 | 492 | $29,456 | -$5,351 | 5.50 | 32 | $5,403 | $229 | -$1,461 | $1,431 |
| ST5_5 & ST5_10 & ST5_20 & ST15_20 | 4 | 509 | $29,418 | -$5,351 | 5.50 | 33 | $5,278 | $103 | -$1,587 | $1,431 |
| ST5_5 & ST5_10 & ST5_50 & ST15_20 | 4 | 495 | $29,213 | -$5,351 | 5.46 | 32 | $5,403 | $229 | -$1,461 | $1,431 |
| ST5_5 & ST5_10 & ST15_20 | 3 | 523 | $28,807 | -$5,351 | 5.38 | 35 | $5,948 | $103 | -$1,587 | $1,431 |
| ST5_20 & ST15_20 | 2 | 567 | $30,892 | -$5,756 | 5.37 | 37 | $4,591 | $103 | -$1,587 | $879 |

Catatan: kombinasi multi-filter dapat memperbaiki March drawdown secara besar,
tetapi trade count turun drastis. Untuk menghindari curve fitting, kandidat
yang lebih sederhana tetap diprioritaskan sebelum kombinasi kompleks.

### 10.4 Keputusan Sementara SuperTrend

Untuk saat ini baseline **tidak diganti**. Baseline tetap `Long only, no ST`
sebagai control. Kandidat yang dibawa ke iterasi berikutnya:

1. `Long only + ST5_50 bullish` sebagai P0 regime-filter candidate.
2. `Long+Short + ST5_50 aligned` sebagai exploratory candidate, bukan prioritas
   utama, karena 30D terakhir masih negatif.

---


## 11. Monte Carlo dan Stress Test

Monte Carlo dilakukan dengan bootstrap dari daily PnL historis. Ini bukan
prediksi masa depan, tetapi stress test distribusi jika pola daily PnL historis
muncul dalam urutan yang berbeda.

| Horizon | Median PnL | P5 PnL | Prob. Akhir Rugi | Median MaxDD | Prob. DD <= -$2k | Prob. Hit +$3k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30D | $641 | -$4,348 | 40.32% | -$2,269 | 57.94% | 28.44% |
| 100D | $1,893 | -$6,990 | 35.46% | -$4,646 | 95.96% | 64.36% |
| 200D | $3,730 | -$8,746 | 31.10% | -$6,568 | 99.86% | 78.14% |

### 11.1 Fan Chart 30D

![Monte Carlo PnL Fan 30D](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/monte_carlo/monte_pnl_fan_30d.png)

### 11.2 Distribusi Final PnL 30D

![Monte Carlo Final PnL CDF 30D](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/monte_carlo/monte_final_pnl_cdf_30d.png)

### 11.3 Max Drawdown 30D

![Monte Carlo MaxDD 30D](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/monte_carlo/monte_maxdd_hist_30d.png)

### 11.4 Fan Chart 100D

![Monte Carlo PnL Fan 100D](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/monte_carlo/monte_pnl_fan_100d.png)

Kesimpulan Monte Carlo: strategi punya upside untuk mencapai +$3,000 dalam
sebagian path 30D, tetapi risiko drawdown terhadap batas -$2,000 tetap perlu
diuji lebih ketat dengan simulator Topstep yang memperhitungkan aturan akun.

---

## 12. Penilaian Risiko

### 12.1 Risiko Drawdown

Max drawdown historis -$12,124 jauh lebih besar daripada MLL
Topstep 50K. Ini tidak otomatis membatalkan strategi, karena evaluasi Topstep
berjalan pada window pendek, tetapi artinya strategi membutuhkan guard dan
monitoring harian.

### 12.2 Risiko No Normal SL

Strategi ini tidak memakai SL normal. Exit loss terjadi lewat time exit.
Konsekuensinya, flash drop atau trend day yang berlawanan bisa menghasilkan
kerugian lebih besar dari target risk teoritis. Catastrophic guard harus
dipilih sebagai layer operasional terpisah.

### 12.3 Risiko Curve Fit

Baseline ini cukup bersih karena hanya memakai OR 15m, long only, TP 2R/time
exit, dan risk $500. Namun pemilihan parameter tetap berasal dari sweep, jadi
forward test diperlukan sebelum dianggap valid.

### 12.4 Risiko Eksekusi Live

Live version harus memastikan:

- M1 candle close sudah final sebelum entry.
- Entry dilakukan pada open M1 berikutnya.
- Jam New York dan daylight saving benar.
- Tidak ada duplicate trade per hari.
- Tidak ada posisi tanpa catastrophic guard.
- Data feed dan broker connection punya heartbeat.

---

## 13. Rekomendasi Sementara

| Area | Rekomendasi |
| --- | --- |
| Baseline research | Pertahankan sebagai control strategy |
| Live trading | Belum live-ready |
| Forward test | Layak dibuat paper/forward-test setelah Topstep simulator selesai |
| ML overlay | Hanya boleh menjadi risk adjuster, bukan filter trade utama dulu |
| Sizing default | Tetap $500 sampai MLL/consistency simulator selesai |
| Guard | Wajib desain catastrophic guard sebelum live |

Rekomendasi utama:

1. Jadikan `rule_based_15m_long_tp2r_eod` sebagai benchmark MNQ ORB.
2. Jangan mengganti baseline dengan ML sebelum ML terbukti memperbaiki risk
   adjusted return terhadap baseline ini.
3. Prioritas berikutnya adalah Topstep-specific simulator: MLL, consistency,
   first +$3,000 path, dan daily loss guard.

---

## 14. Keputusan Sementara

| Area | Status |
| --- | --- |
| Baseline edge | Ada, tetapi tipis |
| 30D Topstep-style potential | Menarik |
| Long-run robustness | Perlu guard dan regime review |
| Live readiness | Belum |
| Model package | Siap sebagai baseline report |

Keputusan sementara: **strategi dipertahankan sebagai baseline MNQ ORB v1**.
Belum ada approval untuk live execution.

---

## 15. Artifact Register

### Model Package

| File | Keterangan |
| --- | --- |
| `README.md` | Model card singkat |
| `REPORT.md` | Laporan utama ini |
| `metrics.json` | Ringkasan metrik machine-readable |
| `manifest.json` | Lineage source/output |
| `charts/equity_curve.png` | Equity curve |
| `charts/drawdown_curve.png` | Drawdown curve |
| `charts/monthly_pnl.png` | Monthly PnL |
| `charts/rolling_windows.png` | Rolling window PnL/DD |
| `charts/trade_pnl_distribution.png` | Distribusi PnL trade |
| `charts/supertrend_variant_equity_curve.png` | Equity curve perbandingan varian ST5_50 |
| `charts/supertrend_variant_drawdown_curve.png` | Drawdown curve perbandingan varian ST5_50 |
| `charts/supertrend_variant_monthly_pnl_2026.png` | Monthly PnL 2026 perbandingan varian ST5_50 |
| `charts/supertrend_variant_rolling_windows.png` | Rolling PnL/DD perbandingan varian ST5_50 |
| `charts/supertrend_variant_trade_pnl_distribution.png` | Distribusi trade PnL perbandingan varian ST5_50 |
| `charts/supertrend_variant_march_2026_equity.png` | Equity khusus March 2026 perbandingan varian ST5_50 |
| `monte_carlo/monte_pnl_fan_30d.png` | Monte Carlo fan chart 30D |
| `monte_carlo/monte_final_pnl_cdf_30d.png` | Monte Carlo final PnL CDF 30D |
| `monte_carlo/monte_maxdd_hist_30d.png` | Monte Carlo MaxDD histogram 30D |
| `monte_carlo/monte_pnl_fan_100d.png` | Monte Carlo fan chart 100D |
| `supertrend_regime_audit.md` | Audit grid SuperTrend 5m/15m ATR 5/10/20/50 |
| `supertrend_filter_candidates.csv` | Semua kandidat kombinasi bullish SuperTrend |
| `supertrend_variant_comparison.md` | Perbandingan baseline, ST5_50, long+short, dan long+short ST aligned |
| `supertrend_variant_comparison.csv` | Tabel machine-readable untuk perbandingan variant ST5_50 |

### Canonical Data

```text
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/events.parquet
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/summary.json
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/manifest.json
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/supertrend_regime_features.parquet
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/supertrend_regime_manifest.json
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/supertrend_variant_comparison_manifest.json
```

---

## 16. Lampiran A - 10 Trade Terakhir

| NY Date | Signal UTC | Exit | Contracts | Net PnL |
| --- | --- | --- | ---: | ---: |
| 2026-05-08 | 2026-05-08 13:46 | TIME_EXIT | 1 | $397 |
| 2026-05-11 | 2026-05-11 15:44 | TIME_EXIT | 1 | -$102 |
| 2026-05-13 | 2026-05-13 14:53 | TP_2R | 2 | $885 |
| 2026-05-14 | 2026-05-14 13:46 | TIME_EXIT | 1 | $98 |
| 2026-05-15 | 2026-05-15 14:15 | TIME_EXIT | 1 | $175 |
| 2026-05-19 | 2026-05-19 17:10 | TIME_EXIT | 1 | -$222 |
| 2026-05-20 | 2026-05-20 13:47 | TIME_EXIT | 1 | $213 |
| 2026-05-21 | 2026-05-21 13:55 | TP_2R | 2 | $979 |
| 2026-05-22 | 2026-05-22 17:30 | TIME_EXIT | 1 | -$175 |
| 2026-05-26 | 2026-05-26 13:46 | TIME_EXIT | 1 | $101 |

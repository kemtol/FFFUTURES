# Strategi NASDAQ Micro Futures Opening Range Breakout Rule-Based Iterasi v1
**Evaluasi Baseline 15m Long TP2R/EOD pada kontrak MNQ**

Tanggal laporan: **2026-06-05**

Model / strategy ID: `rule_based_15m_long_tp2r_eod`

Objective: **Topstep 50K research baseline and regime-filter comparison** -
mencari apakah breakout NASDAQ Micro Futures setelah 15 menit pertama New York
open punya positive expectancy yang cukup untuk menjadi kandidat forward test,
lalu menilai apakah SuperTrend sederhana dapat memperbaiki drawdown tanpa
merusak recent performance.

Audience: trader futures, evaluator internal strategi NASDAQ futures, dan
pembanding untuk overlay machine learning.

---

## 1. Ringkasan Eksekutif

Laporan ini mengevaluasi strategi NASDAQ Micro Futures ORB v1 sebagai
**rule-based research package**. Baseline long-only tetap menjadi control,
tetapi report ini juga memuat comparison terhadap regime filter SuperTrend dan
eksplorasi long+short. Ticker teknis yang digunakan di data dan backtest adalah
`MNQ`, yaitu Micro E-mini Nasdaq-100 futures.

Aturan yang diuji sederhana: ambil posisi long setelah candle M1 pertama close
di atas high opening range 15 menit, entry pada open M1 berikutnya, lalu exit
di TP 2R atau time exit 15:00 New York. Strategi ini tidak memakai normal stop
loss; OR low hanya menjadi referensi sizing.

| Area | Hasil |
| --- | ---: |
| Periode sinyal | 2019-05-06 - 2026-05-26 |
| Baseline total trade | 1,296 |
| Baseline win rate | 56.48% |
| Baseline net PnL | $33,091 |
| Baseline max drawdown | -$12,124 |
| Baseline profit factor | 1.12 |
| Baseline daily Sharpe / Sortino | 0.50 / 0.64 |
| Baseline 30D terakhir | 18 trade, $3,460 PnL, -$551 max DD |

Ringkasan varian utama:

| Variant | Trades | PnL | DD | Ret/DD | Mar 2026 PnL | 30D PnL | Current Use |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Long only, no ST | 1,296 | $33,091 | -$12,124 | 2.73 | -$2,633 | $3,460 | Control baseline |
| Long only, ST5_50 bullish | 964 | $28,199 | -$8,027 | 3.51 | $102 | $1,918 | P0 candidate |
| Long+Short, no ST | 1,767 | $26,501 | -$15,294 | 1.73 | -$1,451 | $839 | Rejected as primary |
| Long+Short, ST5_50 aligned | 1,251 | $36,800 | -$9,099 | 4.04 | $2,336 | -$1,059 | Exploratory only |
| Best P0 short-switch | 1,660 | $42,946 | -$12,792 | 3.36 | -$975 | $4,696 | Enhancement 3 / P1 candidate |

Keputusan sementara dari comparison ini:

- Baseline `Long only, no ST` tetap menjadi **control strategy** karena paling
  mudah diaudit dan menjadi pembanding wajib.
- `Long only + ST5_50 bullish` menjadi **P0 regime-filter candidate**:
  drawdown full-history dan March 2026 membaik dengan hanya satu rule tambahan.
- `Best P0 short-switch` adalah **Enhancement 3** yang saat ini paling kuat
  pada Topstep-style recent window: 30D PnL dan March 2026 membaik terhadap
  baseline, tetapi full-history DD sedikit lebih berat.
- `Long+Short, no ST` tidak dipromosikan karena short side mentah menambah
  frekuensi tetapi menurunkan kualitas risk-adjusted.
- `Long+Short + ST5_50 aligned` tetap ditrack sebagai exploratory variant:
  full-history dan March terlihat bagus, tetapi 30D terakhir negatif.


**Kesimpulan utama:** strategi ini belum boleh dibaca sebagai satu final live
strategy. Baseline membuktikan ada continuation edge, terutama pada 30D
terakhir, tetapi long-run PF masih tipis dan max drawdown historis terlalu
besar untuk langsung masuk Topstep live. SuperTrend `ST5_50` memberi perbaikan
drawdown yang jelas, khususnya pada March 2026, namun menurunkan 30D PnL.
Enhancement 3 melalui Best P0 short-switch memperbaiki March 2026 dan 30D
terakhir terhadap baseline, tetapi full-history DD sedikit lebih berat. Maka
keputusan institusional saat ini adalah: baseline tetap control, Best P0
short-switch masuk P1 simulator candidate, ST5_50 menjadi simpler fallback, dan
semua variant perlu Topstep MLL/consistency simulator sebelum forward execution.

---

## 2. Latar Belakang Strategi

Opening Range Breakout berangkat dari hipotesis bahwa rentang harga pada awal
sesi New York menyimpan informasi tentang imbalance intraday. Untuk Nasdaq
futures, tekanan order setelah cash open sering menjadi penentu arah sesi.
Strategi ini mencari continuation setelah harga keluar dari opening range,
bukan mean reversion intraday.

### 2.1 Research Problem

Target riset bukan hanya mencari total PnL tertinggi. Untuk konteks Topstep 50K,
strategi harus menjawab beberapa pertanyaan praktis:

1. Apakah ORB NASDAQ Micro Futures 15m punya positive expectancy setelah biaya dan slippage?
2. Apakah edge cukup aktif untuk window evaluasi sekitar 30 hari?
3. Apakah drawdown masih masuk akal terhadap MLL dan consistency rule?
4. Apakah filter sederhana dapat mengurangi bulan buruk seperti March 2026
   tanpa menghapus trade terbaik pada April-May 2026?
5. Apakah sisi short menambah edge atau hanya menambah noise/frequency?

### 2.2 Why Baseline First

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

Baseline long-only juga berfungsi sebagai control: setiap filter, ML model,
atau long+short extension harus mengalahkan baseline pada risk-adjusted metrics,
bukan hanya menaikkan satu angka PnL.

### 2.3 Why SuperTrend Was Added To The Audit

March 2026 menunjukkan kelemahan utama baseline: continuation long-only bisa
terjebak pada regime yang tidak mendukung breakout. SuperTrend diuji sebagai
regime filter karena:

- Rule-nya eksplisit dan mudah diaudit.
- Bisa dihitung dari bar yang sudah close, sehingga no-lookahead bisa digate.
- Mewakili trend state tanpa langsung menjadi model ML.
- Cocok sebagai risk filter sebelum masuk ke probability sizing.

Audit menghitung ST 5m/15m dengan ATR 5/10/20/50. Kandidat paling sederhana
yang muncul adalah `ST5_50`: long breakout hanya diambil saat ST5_50 bullish.

### 2.4 Why Long+Short Was Tested

Long+short diuji karena breakout bawah secara teori bisa memberi tambahan
frequency. Namun hasil awal menunjukkan short mentah tidak otomatis punya edge.
Ketika short disejajarkan dengan ST5_50 bearish, full-history membaik, tetapi
recent 30D memburuk. Karena itu long+short belum dipromosikan; ia tetap menjadi
exploratory branch yang perlu investigasi lanjutan.

### 2.5 Methodology Guardrails

Semua angka dalam report ini harus dibaca dengan guardrail berikut:

- Entry memakai signal close M1, lalu entry di open M1 berikutnya.
- SuperTrend feature hanya boleh memakai timestamp fitur `<= signal_ts`.
- Biaya TopstepX MNQ dan slippage sudah masuk.
- Baseline dan varian ST adalah rule-based, bukan ML.
- Laporan ini research-only; belum live-ready.

---

## 3. Data Lineage & Control Gates

Ini bagian paling penting secara governance: **bronze/L0 data salah berarti
semua angka PnL, drawdown, chart, dan kesimpulan report ini ikut salah**.
Karena itu report ini harus dibaca dari bawah ke atas: L0 raw integrity,
L1 feature/context integrity, L2 event integrity, lalu baru performance.

### 3.1 Lineage

```text
Databento MNQ M1 + yfinance recent append
  -> data/Level_0_Raw/MNQ_1m.duckdb                 # bronze/raw canonical M1
  -> data/Level_1_Features/mnq/ORB/context.parquet  # M1 NY-session context + quality flags
  -> data/Level_2_Datamart/mnq/ORB/sweeps           # ORB grid events
  -> data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod
  -> model/MNQ/ORB/rule_based_15m_long_tp2r_eod
```

Baseline control report ini memakai frozen L2 event package
`rule_based_15m_long_tp2r_eod/events.parquet`. SuperTrend dan short-switch
adalah audit/variant layer yang menempel ke baseline/sweep events, bukan data
source baru.

### 3.2 Gate Summary

| Layer | Artifact | Control Result |
| --- | --- | --- |
| Bronze / L0 M1 OHLCV | `data/Level_0_Raw/MNQ_1m.duckdb` | 2,487,923 rows, 2019-05-05 22:03:00 to 2026-05-29 20:59:00; hard integrity PASS; duplicates 0; null OHLCV 0; bad OHLC 0; negative volume 0 |
| L0 continuity | `data/Level_0_Raw/MNQ_1m_continuity_report.json` | PASS_WITH_GAPS_REQUIRING_L1_QUARANTINE; gaps >60s 5,491; max gap 284,460s; downstream rule: quarantine gap bars, do not train across gaps |
| Recent yfinance append | `data/Level_0_Raw/MNQ_1m_yfinance_append_manifest.json` | 8,881 appended/replaced rows, 2026-05-20 04:09:00 to 2026-05-29 20:59:00; post-append duplicates 0 |
| Derived L0 5m/15m | `MNQ_5m.duckdb`, `MNQ_15m.duckdb` | right-labeled, left-closed from M1; 5m rows 498,458, 15m rows 166,200; duplicate timestamps 5m/15m = 0/0 |
| yfinance timeframe parity | `data/Level_0_Raw/MNQ_yfinance_timeframe_parity_report.json` | PASS; max mismatch rate 0.00%; latest incomplete bars excluded 3 |
| L1 intraday context | `data/Level_1_Features/mnq/ORB/context.parquet` | PASS; 2,487,265 rows, 18 columns; 2,199 NY days; OR-complete days 1,815; duplicate timestamps 0; bad OHLC rows 0; required hard nulls 0 |
| L1 ORB leakage guards | `data/Level_1_Features/mnq/ORB/l1_audit.json` | pre-OR rows with OR values 0; eligible rows before OR end 0; eligible bad-quality rows 0; complete OR days wrong bar count 0 |
| Daily confluence features | `data/Level_1_Features/mnq/ORB/daily_confluence.parquet` | PASS; 2,199 rows, 29 features; feature nulls 0; lookahead violations 0; contract: external daily feature date < MNQ trade date |
| L2 baseline events | `data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/events.parquet` | 1,296 rows, 38 columns; required nulls 0; duplicate NY dates 0; entry<=signal rows 0; exit<entry rows 0 |
| Frozen package gate | `data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/package_gate.json` | PASS; events 1,296; failures 0; warnings 2 |
| SuperTrend feature attach | `supertrend_regime_features.parquet` | lookahead violations 0; variant attach lookahead 0; max feature lag 14.0 minutes |

### 3.3 Feature Dataset Inventory

| Layer | Dataset | Content | Shape | Role |
| --- | --- | --- | --- | --- |
| Bronze/L0 raw | `MNQ_1m.duckdb` | OHLCV M1, source symbol, volume | 2,487,923 rows | Raw market bars; if this layer is wrong, all reports are invalid |
| L1 intraday context | `context.parquet` | OHLCV, NY date/time, `minutes_from_open`, quality flags, OR context | 2,487,265 rows / 18 cols | Research context; sweep recomputes 10/15/20/30m OR only from quality bars |
| L1 daily confluence | `daily_confluence.parquet` | SPY, QQQ, VIX, TNX, DXY prior-day features | 2,199 rows / 29 features | Not used by this rule-based report, but available for ML overlays with strict D-1 contract |
| L2 sweep | `sweeps/sweep_events.parquet` | ORB parameter grid events | 159,294 rows | Parent grid for selecting current baseline candidate |
| L2 baseline package | `rule_based_15m_long_tp2r_eod/events.parquet` | Executed trade events, sizing, cost-adjusted PnL | 1,296 rows / 38 cols | Current benchmark/control strategy |
| L2 ST feature package | `supertrend_regime_features.parquet` | ST5/ST15 ATR 5/10/20/50 values, dirs, distances, lags | 1,296 rows / 90 feature cols added | Regime-filter audit only; attached with feature timestamp <= signal timestamp |

### 3.4 Lookahead Contract

| Component | Contract |
| --- | --- |
| ORB baseline | Signal is M1 candle close; execution is next M1 open, so `entry_ts > signal_ts` is mandatory. |
| Opening range | OR high/low use only completed bars inside the first N minutes after 09:30 NY. |
| Intraday quality | Trade entry/exit scans use `bar_data_quality_ok`; bars containing source gaps are excluded from decision scanning. |
| SuperTrend attach | Feature timestamp must be `<= signal_ts`; current attached lookahead violations = 0. |
| Daily confluence | External daily features use date `< ny_date`; current daily confluence lookahead violations = 0. |
| Labels/PnL | Exit, PnL, R multiple, and realized outcome are labels/evaluation fields, not allowed as pre-trade features. |

### 3.5 Current Data Risk Read

- L0 hard integrity is clean: no duplicate timestamps, null OHLCV, bad OHLC, or
  negative volume rows in the latest audit.
- L0 continuity is **not perfectly gap-free**. The current status is
  `PASS_WITH_GAPS_REQUIRING_L1_QUARANTINE`, so the correct interpretation is
  not "data has no gaps", but "known gaps are flagged and downstream builders
  must not train or trade through bad bars".
- L1 context audit passes and explicitly checks that OR values do not appear on
  pre-OR rows, no eligible rows exist before OR end, and eligible rows are not
  bad-quality bars.
- L2 baseline events pass event-level controls used by this report: no required
  nulls, no duplicate NY dates, entry after signal, and exit after entry.
- Any future ML model must promote these gates to hard blockers. A model trained
  on failed L0/L1/L2 gates is invalid even if its backtest looks profitable.

---

## 4. Konteks Strategy Family

Report ini tidak lagi hanya berisi satu baseline long-only. Scope saat ini
adalah **family of rule-based ORB variants** untuk NASDAQ Micro Futures, dengan
baseline sebagai control dan beberapa branch sebagai kandidat riset.

### 4.1 Common Contract

| Field | Value |
| --- | --- |
| Instrument | NASDAQ Micro Futures (`MNQ`) |
| Session | New York regular session |
| Source grain | Right-labeled M1 bars |
| Opening range | First 15 minutes after 09:30 NY |
| Entry timing | Signal after M1 close, execution on next M1 open |
| Primary exit clock | 15:00 NY EOD/time exit |
| Cost model | TopstepX MNQ commission + 1 tick slippage per side |
| Target risk | $500 |
| Baseline max trades | 1 sequence per NY session |
| Research status | Not live-ready |

### 4.2 Variant Map

| Variant | Role | Direction Logic | Regime Filter | Exit Logic | Current Status |
| --- | --- | --- | --- | --- | --- |
| Long only, no ST | Control baseline | First M1 close above OR high | None | TP 2R or 15:00 NY | Keep as benchmark |
| Long only + ST5_50 | P0 regime-filter candidate | Same as baseline | Require ST5_50 bullish at signal close | TP 2R or 15:00 NY | Best simple regime filter |
| Long+Short, no ST | Rejected as primary | First breakout either OR high or OR low | None | TP 2R or 15:00 NY | Adds frequency but weak risk-adjusted quality |
| Long+Short + ST5_50 aligned | Exploratory | Long with bullish ST, short with bearish ST | ST5_50 aligned by side | TP 2R or 15:00 NY | Good March, weak recent 30D |
| Short switch to long | Research branch | Short if OR low breaks first; switch to long if OR high reclaimed | None in current test | Short TP 1R/1.5R/2R, switch, or EOD | TP 2R best, not promoted |
| Best P0 short-switch | Enhancement 3 / P1 candidate | Long-first baseline, plus filtered short-switch branch | Short side requires `ST5_20` bearish; short risk $350; switch-long risk $750; guard 10:30 | Short TP 2R, switch to long on OR high reclaim, or EOD | Best recent-window improvement so far; needs Topstep simulator |

### 4.3 Rule-Based Definition

Semua variant di report ini masih **rule-based**, bukan ML. Keputusan entry,
exit, switch, dan filter ditentukan oleh aturan eksplisit. Belum ada model
probabilitas yang menentukan trade size, trade/no-trade, atau direction.

### 4.4 Current Promotion Hierarchy

1. **Benchmark:** `Long only, no ST`.
2. **Enhancement 3 / P1 candidate:** `Best P0 short-switch`
   (`p0_st5_20_bearish_sr350_lr750_buf0_tg1030`).
3. **P0 regime-filter candidate:** `Long only + ST5_50 bullish`.
4. **Watchlist:** `Short switch to long, short TP 2R` sebelum P0 tuning.
5. **Exploratory only:** `Long+Short + ST5_50 aligned`.
6. **Not promoted:** `Long+Short, no ST`.

---

## 5. Sizing dan Risk Model

Semua varian di report ini memakai target risk dollar tetap sebagai sizing
anchor. Target risk bukan normal stop-loss order; ia hanya menentukan jumlah
kontrak berdasarkan jarak entry ke referensi opening range.

```text
contracts_float = target_risk_usd / risk_per_contract_usd
contracts_used = floor(contracts_float), minimum 1 contract
```

Dengan `target_risk_usd = $500`, jumlah kontrak otomatis turun saat OR/risk
melebar dan naik saat risk menyempit. Karena kontrak harus integer, actual risk
tidak selalu tepat $500.

Referensi risk per family:

| Family | Risk Reference | Exit |
| --- | --- | --- |
| Long only baseline | Entry minus OR low | TP 2R atau 15:00 NY |
| Long only + ST5_50 | Entry minus OR low | TP 2R atau 15:00 NY |
| Short continuation | OR high minus entry | TP 2R atau 15:00 NY |
| Short switch to long | Short leg memakai OR high; switched long memakai OR low | Short TP/switch/EOD, lalu long TP 2R/EOD |

Catatan penting: OR high/low **bukan** normal stop loss strategi. Exit loss
utama tetap time exit, sehingga live version tetap membutuhkan catastrophic
guard terpisah.

---

## 6. Baseline Control - Equity, Drawdown, Monthly PnL

Section ini hanya untuk baseline control `Long only, no ST`. Tujuannya adalah
menyediakan benchmark bersih sebelum membaca audit SuperTrend dan short-switch
di section 11-12.

### 6.1 Equity Curve

![Equity Curve](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/equity_curve.png)

Equity curve baseline menunjukkan PnL positif secara historis, tetapi jalurnya
tidak linear. Ada fase panjang yang relatif datar dan beberapa periode drawdown
besar.

### 6.2 Drawdown

![Drawdown Curve](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/drawdown_curve.png)

Drawdown maksimum historis sebesar -$12,124. Ini jauh lebih
besar daripada batas MLL Topstep 50K, sehingga evaluasi live tidak boleh hanya
mengandalkan total PnL historis.

### 6.3 Monthly PnL

![Monthly PnL](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/monthly_pnl.png)

Grafik bulanan baseline membantu melihat bahwa strategi tidak menghasilkan
distribusi profit yang stabil setiap bulan. Ada bulan kuat, bulan kosong, dan
bulan rugi.

### 6.4 Distribusi PnL Per Trade

![Trade PnL Distribution](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/trade_pnl_distribution.png)

Pada baseline, rata-rata loss per trade masih lebih besar daripada rata-rata
win. Edge muncul dari kombinasi win rate 56.48%, sizing, dan beberapa periode
momentum yang produktif.

---

## 7. Baseline Control - Performance Card dan Variant Snapshot

Section 7.1 adalah metric card baseline `Long only, no ST`. Ini bukan metric
untuk seluruh strategy family. Cross-variant context langsung ditaruh di
section 7.2 agar baseline, ST filter, dan short-switch tidak tercampur.

### 7.1 Baseline Control Metrics

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

### 7.2 Cross-Variant Metric Snapshot

| Variant | Role | Trades | PF | Full PnL | Max DD | Mar 2026 PnL | 30D PnL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Long only, no ST | Control | 1,296 | 1.12 | $33,091 | -$12,124 | -$2,633 | $3,460 |
| Long only, ST5_50 bullish | P0 candidate | 964 | 1.14 | $28,199 | -$8,027 | $102 | $1,918 |
| Long+Short, no ST | Rejected | 1,767 | 1.06 | $26,501 | -$15,294 | -$1,451 | $839 |
| Long+Short, ST5_50 aligned | Exploratory | 1,251 | 1.13 | $36,800 | -$9,099 | $2,336 | -$1,059 |
| Short switch to long, short TP 2R | Watchlist | 1,767 | 1.09 | $37,731 | -$12,715 | -$2,074 | $1,515 |
| Best P0 short-switch | Enhancement 3 | 1,660 | 1.12 | $42,946 | -$12,792 | -$975 | $4,696 |

Reading note:

- Baseline remains the mandatory control, but Best P0 short-switch now has the
  strongest recent-window improvement in this package.
- `Long only + ST5_50` improves full-history drawdown and March 2026, but gives
  up some recent upside.
- Short-switch TP2R before P0 tuning improved full-history PnL, but did not
  beat baseline on March/30D. The P0 tuned variant fixes that recent-window
  weakness, while still requiring Topstep-specific simulator validation.

---

## 8. Cost Model

| Cost | Value |
| --- | ---: |
| Commission + fees | $1.24 RT / contract |
| Slippage | 1 tick per side |
| Modeled slippage | $1.00 RT / contract |
| Total commission paid | $5,590 |
| Total modeled slippage | $4,508 |

Biaya ini dipakai konsisten untuk baseline-control dan audit varian yang dibuat
dalam package ini. `pnl_net_usd` sudah memasukkan TopstepX MNQ, yaitu kontrak
Micro E-mini Nasdaq-100 futures, $1.24 round-turn per contract dan modeled
slippage 1 tick per side.

---

## 9. Baseline Control - Daily Quality

Daily quality di section ini hanya untuk baseline-control. Sharpe and Sortino
are computed from daily dollar PnL over NASDAQ Micro Futures NY session days,
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

Variant daily quality belum dijadikan decision metric utama karena varian ST dan
short-switch masih perlu Topstep-specific simulator yang sama sebelum promosi.

---

## 10. Baseline Control - Rolling Window Terakhir

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

Interpretasi baseline:

- 30D terakhir adalah bagian paling menarik: 18 trade dan $3,460 PnL.
- 5D dan 10D masih terlalu pendek untuk menjadi bukti edge.
- 100D dan 200D tetap positif, tetapi DD historisnya mulai berat untuk Topstep.
- Rolling comparison untuk varian ST dan short-switch ada di section 11-12,
  bukan di chart baseline ini.

---

## 11. SuperTrend Regime Filter Audit

SuperTrend audit ditambahkan untuk menjawab apakah drawdown March 2026 bisa
dikurangi dengan regime filter sederhana, tanpa langsung mengganti baseline.
Semua fitur dihitung dari bar yang sudah close dan di-join ke trade event
dengan rule `feature_ts <= signal_ts`.

### 11.1 Data Integrity

| Check | Value |
| --- | ---: |
| Feature family | `ST5_5`, `ST5_10`, `ST5_20`, `ST5_50`, `ST15_5`, `ST15_10`, `ST15_20`, `ST15_50` |
| SuperTrend factor | 4.00 |
| Direction convention | `-1 = bullish/up`, `+1 = bearish/down` |
| Join rule | Latest completed feature timestamp `<= signal_ts` |
| Lookahead violations | 0 |
| Max feature lag | 14 menit |


### 11.2 Perbandingan Variant Utama

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


### 11.3 Kandidat Kombinasi SuperTrend

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

### 11.4 Keputusan Sementara SuperTrend

Untuk saat ini baseline **tidak diganti**. Baseline tetap `Long only, no ST`
sebagai control. Kandidat yang dibawa ke iterasi berikutnya:

1. `Long only + ST5_50 bullish` sebagai P0 regime-filter candidate.
2. `Long+Short + ST5_50 aligned` sebagai exploratory candidate, bukan prioritas
   utama, karena 30D terakhir masih negatif.

---


## 12. Short Breakout Switch-To-Long Audit

Section ini menguji definisi short yang asimetris terhadap long. Karena NASDAQ
secara natural lebih long-biased, short tidak diperlakukan sebagai mirror
strategy. Jika OR low break lebih dulu, strategy boleh masuk short; tetapi jika
harga close kembali di atas OR high, short ditutup dan posisi dibalik menjadi
long pada open M1 berikutnya.

### 12.1 Methodology

| Field | Value |
| --- | --- |
| Short entry | First M1 close below OR low |
| Short exit | TP 1R / 1.5R / 2R, OR switch to long, OR 15:00 NY EOD |
| Switch trigger | First M1 close above OR high while short is active |
| Switch execution | Close short and open long at next M1 open |
| Long after switch | Baseline long TP 2R or 15:00 NY EOD |
| Anchor | 2026-05-28T01:53:00+00:00 |

### 12.2 Visual Audit

#### Equity Curve

![Short Switch Equity](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/short_reversal_switch_equity_curve.png)

#### Drawdown Curve

![Short Switch Drawdown](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/short_reversal_switch_drawdown_curve.png)

#### Last 30D Equity

![Short Switch Last 30D](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/short_reversal_switch_last30_equity.png)

### 12.3 Summary

| Variant | Trades | WR | PnL | DD | Ret/DD | Short-first | Switches | Short PnL | Jan-May PnL | Mar PnL | 30D PnL | 30D DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Long only, no ST | 1,296 | 56.48% | $33,091 | -$12,124 | 2.73 | 0 | 0 | $0 | $6,096 | -$2,633 | $3,460 | -$551 |
| Long+Short first breakout, no switch | 1,767 | 53.03% | $26,501 | -$15,294 | 1.73 | 862 | 0 | $0 | $4,072 | -$1,451 | $839 | -$1,636 |
| Short switch to long, short TP 1.5R | 1,767 | 54.90% | $26,124 | -$16,047 | 1.63 | 862 | 375 | -$7,598 | $6,592 | -$2,085 | $1,323 | -$886 |
| Short switch to long, short TP 1R | 1,767 | 57.89% | $32,242 | -$13,722 | 2.35 | 862 | 340 | -$607 | $4,270 | -$2,480 | $973 | -$886 |
| Short switch to long, short TP 2R | 1,767 | 53.99% | $37,731 | -$12,715 | 2.97 | 862 | 382 | $3,671 | $7,085 | -$2,074 | $1,515 | -$886 |

### 12.4 Current Read

Di antara varian short-switch, short TP 2R adalah yang paling kuat: total PnL
dan return/DD terbaik, serta short leg full-history positif. Namun ia masih
belum mengalahkan baseline pada window 30D terakhir dan max drawdown-nya masih
sedikit lebih berat dari baseline. Jadi short-switch TP 2R layak masuk watchlist
sebagai research branch, tetapi belum menggantikan long-only baseline.

### 12.5 P0 Short-Switch TP2R Optimization Sweep

P0 sweep ini memperbaiki branch `Short switch to long, short TP 2R` dengan
knob yang masih rule-based: short-side SuperTrend filter, asymmetric short risk,
switch-long risk, dan switch trigger buffer. Long-first breakout tetap memakai
baseline risk $500. Jika short breakout ditolak filter, hari tersebut masih
boleh mengambil later long breakout.

| Field | Value |
| --- | --- |
| Variants evaluated | 216 plus baseline |
| Short filters | `none`, `st5_50_bearish`, `st5_20_bearish`, `st5_50_and_st15_20_bearish` |
| Short risk grid | $250, $350, $500 |
| Switch long risk grid | $500, $750 |
| Switch buffers | `0`, `2ticks`, `0.25r` |
| Short time guard grid | `none, 10:30, 11:00` |
| Lookahead violations | 0 |
| Max ST feature lag | 14.00 minutes |

#### Visual Comparison Enhancement 3

Gambar ini membandingkan tiga state utama: baseline control, short-switch TP2R
lama, dan best P0 enhancement.

##### Equity Curve

![P0 Equity Comparison](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/short_switch_tp2r_p0_equity_curve.png)

##### Drawdown Curve

![P0 Drawdown Comparison](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/short_switch_tp2r_p0_drawdown_curve.png)

##### Monthly PnL 2026

![P0 Monthly PnL 2026](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/short_switch_tp2r_p0_monthly_pnl_2026.png)

##### Rolling Window PnL/DD

![P0 Rolling Windows](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/short_switch_tp2r_p0_rolling_windows.png)

##### Last 30D Equity

![P0 Last 30D Equity](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/charts/short_switch_tp2r_p0_last30_equity.png)

| Candidate | Short Filter | Short Risk | Switch Long Risk | Buffer | Trades | PnL | DD | Ret/DD | Mar PnL | 30D Trades | 30D PnL | 30D DD |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Long only baseline | baseline | - | - | baseline | 1,296 | $33,091 | -$12,124 | 2.73 | -$2,633 | 18 | $3,460 | -$551 |
| Existing short-switch TP2R equivalent | none | $500 | $500 | 0 | 1,767 | $37,731 | -$12,715 | 2.97 | -$2,074 | 21 | $1,515 | -$886 |
| Best P0 score | st5_20_bearish | $350 | $750 | 0 | 1,660 | $42,946 | -$12,792 | 3.36 | -$975 | 16 | $4,696 | -$551 |
| Best raw 30D PnL | st5_50_bearish | $350 | $750 | 0 | 1,711 | $38,950 | -$14,135 | 2.76 | -$975 | 15 | $4,724 | -$551 |
| Best full Ret/DD | st5_20_bearish | $500 | $500 | 0 | 1,706 | $50,558 | -$10,410 | 4.86 | -$1,366 | 20 | $2,649 | -$886 |

Top P0-score candidates:

| Candidate | Short Filter | Short Risk | Switch Long Risk | Buffer | Trades | PnL | DD | Ret/DD | Mar PnL | 30D Trades | 30D PnL | 30D DD |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p0_st5_20_bearish_sr350_lr750_buf0_tg1030 | st5_20_bearish | $350 | $750 | 0 | 1,660 | $42,946 | -$12,792 | 3.36 | -$975 | 16 | $4,696 | -$551 |
| p0_st5_20_bearish_sr350_lr750_buf0_tg1100 | st5_20_bearish | $350 | $750 | 0 | 1,692 | $42,884 | -$13,139 | 3.26 | -$975 | 16 | $4,696 | -$551 |
| p0_st5_50_bearish_sr350_lr750_buf0_tg1100 | st5_50_bearish | $350 | $750 | 0 | 1,694 | $41,679 | -$13,055 | 3.19 | -$975 | 15 | $4,724 | -$551 |
| p0_st5_20_bearish_sr350_lr750_buf0_tgnone | st5_20_bearish | $350 | $750 | 0 | 1,713 | $40,368 | -$14,158 | 2.85 | -$975 | 16 | $4,696 | -$551 |
| p0_st5_50_bearish_sr350_lr750_buf0_tg1030 | st5_50_bearish | $350 | $750 | 0 | 1,666 | $40,951 | -$12,889 | 3.18 | -$975 | 16 | $4,622 | -$551 |
| p0_st5_20_bearish_sr350_lr750_buf2ticks_tg1030 | st5_20_bearish | $350 | $750 | 2ticks | 1,660 | $39,762 | -$14,025 | 2.84 | -$975 | 16 | $4,696 | -$551 |
| p0_st5_20_bearish_sr350_lr750_buf2ticks_tg1100 | st5_20_bearish | $350 | $750 | 2ticks | 1,692 | $39,674 | -$14,372 | 2.76 | -$975 | 16 | $4,696 | -$551 |
| p0_st5_50_bearish_sr350_lr750_buf0_tgnone | st5_50_bearish | $350 | $750 | 0 | 1,711 | $38,950 | -$14,135 | 2.76 | -$975 | 15 | $4,724 | -$551 |
| p0_st5_50_bearish_sr350_lr750_buf2ticks_tg1100 | st5_50_bearish | $350 | $750 | 2ticks | 1,694 | $38,474 | -$14,370 | 2.68 | -$975 | 15 | $4,724 | -$551 |
| p0_st5_20_bearish_sr500_lr750_buf0_tg1030 | st5_20_bearish | $500 | $750 | 0 | 1,706 | $51,410 | -$13,134 | 3.91 | -$1,272 | 20 | $3,789 | -$886 |

Current read:

- Best P0-score candidate: `p0_st5_20_bearish_sr350_lr750_buf0_tg1030`.
- 30D PnL membaik dari $3,460
  menjadi $4,696, dengan 30D DD tetap
  -$551.
- March 2026 membaik dari -$2,633
  menjadi -$975.
- Full-history PnL naik, tetapi full-history DD memburuk dari
  -$12,124 menjadi -$12,792.
  Jadi P0 result **membaik untuk Topstep-style recent window**, tetapi belum
  boleh dipromosikan sebelum P1 Topstep simulator dan time-guard sweep.

Artifact P0:

- `short_switch_tp2r_p0_sweep.md`
- `short_switch_tp2r_p0_full_report.md`
- `short_switch_tp2r_p0_sweep.csv`
- `short_switch_tp2r_p0_best_events.csv`
- `short_switch_tp2r_p0_best_legs.csv`
- `short_switch_tp2r_p0_best_yearly.csv`
- `short_switch_tp2r_p0_best_monthly.csv`
- `data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/short_switch_tp2r_p0_sweep_manifest.json`


---


## 13. Baseline Control - Monte Carlo dan Stress Test

Monte Carlo di section ini hanya memakai daily PnL baseline-control. Ini bukan
prediksi masa depan, dan belum boleh dibaca sebagai Monte Carlo untuk ST5_50
atau short-switch. Fungsinya adalah stress test distribusi baseline jika pola
daily PnL historis muncul dalam urutan yang berbeda.

| Horizon | Median PnL | P5 PnL | Prob. Akhir Rugi | Median MaxDD | Prob. DD <= -$2k | Prob. Hit +$3k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30D | $641 | -$4,348 | 40.32% | -$2,269 | 57.94% | 28.44% |
| 100D | $1,893 | -$6,990 | 35.46% | -$4,646 | 95.96% | 64.36% |
| 200D | $3,730 | -$8,746 | 31.10% | -$6,568 | 99.86% | 78.14% |

### 13.1 Fan Chart 30D

![Monte Carlo PnL Fan 30D](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/monte_carlo/monte_pnl_fan_30d.png)

### 13.2 Distribusi Final PnL 30D

![Monte Carlo Final PnL CDF 30D](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/monte_carlo/monte_final_pnl_cdf_30d.png)

### 13.3 Max Drawdown 30D

![Monte Carlo MaxDD 30D](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/monte_carlo/monte_maxdd_hist_30d.png)

### 13.4 Fan Chart 100D

![Monte Carlo PnL Fan 100D](https://raw.githubusercontent.com/kemtol/FFFUTURES/main/model/MNQ/ORB/rule_based_15m_long_tp2r_eod/monte_carlo/monte_pnl_fan_100d.png)

Kesimpulan Monte Carlo baseline: strategi punya upside untuk mencapai +$3,000
dalam sebagian path 30D, tetapi risiko drawdown terhadap batas -$2,000 tetap
perlu diuji lebih ketat dengan simulator Topstep yang memperhitungkan aturan
akun. Setelah simulator itu ada, baseline, ST5_50, short-switch TP2R lama, dan
Best P0 short-switch harus dibandingkan ulang dengan metodologi yang sama.

---

## 14. Penilaian Risiko

### 14.1 Risiko Drawdown

Baseline max drawdown historis -$12,124 jauh lebih besar
daripada MLL Topstep 50K. ST5_50 menurunkan drawdown full-history, tetapi belum
menghapus risiko MLL karena window 30D dan intraday path tetap harus
disimulasikan. Ini tidak otomatis membatalkan strategi, tetapi semua varian
membutuhkan guard dan monitoring harian.

### 14.2 Risiko No Normal SL

Strategi ini tidak memakai SL normal. Exit loss terjadi lewat time exit.
Konsekuensinya, flash drop atau trend day yang berlawanan bisa menghasilkan
kerugian lebih besar dari target risk teoritis. Catastrophic guard harus
dipilih sebagai layer operasional terpisah.

### 14.3 Risiko Curve Fit

Baseline cukup bersih karena hanya memakai OR 15m, long only, TP 2R/time exit,
dan risk $500. Risiko curve fit naik pada kombinasi multi-SuperTrend dan
short-switch karena jumlah pilihan bertambah. Best P0 short-switch adalah hasil
grid 216 varian, sehingga harus naik ke P1 simulator/robustness check sebelum
dipakai live. `ST5_50` tetap dipertahankan sebagai fallback sederhana karena
hanya menambah satu filter.

### 14.4 Risiko Eksekusi Live

Live version harus memastikan:

- M1 candle close sudah final sebelum entry.
- Entry dilakukan pada open M1 berikutnya.
- Jam New York dan daylight saving benar.
- Tidak ada duplicate trade per hari.
- Tidak ada posisi tanpa catastrophic guard.
- Data feed dan broker connection punya heartbeat.

---

## 15. Rekomendasi Sementara

| Area | Rekomendasi |
| --- | --- |
| Baseline control | Pertahankan `Long only, no ST` sebagai benchmark wajib |
| Enhancement 3 | Bawa `p0_st5_20_bearish_sr350_lr750_buf0_tg1030` ke P1 Topstep simulator |
| P0 regime filter | Tetap track `Long only + ST5_50 bullish` sebagai simpler fallback |
| Short branch | Short-switch TP2R lama tetap baseline pembanding untuk P0 tuned branch |
| Live trading | Belum live-ready |
| Forward test | Baru layak paper/forward-test setelah simulator MLL/consistency selesai |
| ML overlay | Hanya boleh menjadi risk adjuster, bukan filter trade utama dulu |
| Sizing policy | Baseline long-first tetap $500; Best P0 memakai short $350 dan switch-long $750, frozen sampai simulator selesai |
| Guard | Wajib desain catastrophic guard sebelum live |

Rekomendasi utama:

1. Kunci baseline sebagai control, bukan final live strategy.
2. Bandingkan baseline vs `Long only + ST5_50` vs `Short switch TP2R` vs
   `Best P0 short-switch` memakai Topstep-specific simulator yang sama.
3. Jangan mengganti baseline dengan ML sebelum ML terbukti memperbaiki
   risk-adjusted return dan sizing decision terhadap control ini.
4. Prioritas berikutnya adalah simulator: MLL, consistency, first +$3,000 path,
   daily loss guard, dan catastrophic guard.

---

## 16. Keputusan Sementara

| Area | Status |
| --- | --- |
| Strategy family | Rule-based ORB research package |
| Baseline edge | Ada, tetapi PF masih tipis |
| Enhancement 3 / P1 candidate | `p0_st5_20_bearish_sr350_lr750_buf0_tg1030` |
| P0 regime-filter candidate | `Long only + ST5_50 bullish` |
| Watchlist | `Short switch to long, short TP 2R` sebelum P0 tuning |
| 30D Topstep-style potential | P0 short-switch paling menarik sejauh ini, tetapi belum cukup tanpa simulator |
| Long-run robustness | Perlu guard, regime review, dan Topstep path sim |
| Next decision gate | P1 Topstep simulator: MLL, consistency, daily loss, first +$3,000 path, catastrophic guard |
| Live readiness | Belum |
| Model package | Siap sebagai institutional-style research report |

Keputusan sementara: **package ini dipertahankan sebagai NASDAQ Micro Futures
ORB rule-based strategy family v1**. Baseline tetap control, Best P0
short-switch menjadi Enhancement 3 / P1 candidate, ST5_50 menjadi fallback
regime-filter candidate, dan belum ada approval untuk live execution.

---

## 17. Artifact Register

### 17.1 Model Package

| File | Keterangan |
| --- | --- |
| `README.md` | Model card singkat |
| `REPORT.md` | Laporan utama ini |
| `metrics.json` | Ringkasan metrik machine-readable |
| `manifest.json` | Lineage source/output |
| `charts/equity_curve.png` | Baseline-control equity curve |
| `charts/drawdown_curve.png` | Baseline-control drawdown curve |
| `charts/monthly_pnl.png` | Baseline-control monthly PnL |
| `charts/rolling_windows.png` | Baseline-control rolling window PnL/DD |
| `charts/trade_pnl_distribution.png` | Baseline-control distribusi PnL trade |
| `charts/supertrend_variant_equity_curve.png` | Equity curve perbandingan varian ST5_50 |
| `charts/supertrend_variant_drawdown_curve.png` | Drawdown curve perbandingan varian ST5_50 |
| `charts/supertrend_variant_monthly_pnl_2026.png` | Monthly PnL 2026 perbandingan varian ST5_50 |
| `charts/supertrend_variant_rolling_windows.png` | Rolling PnL/DD perbandingan varian ST5_50 |
| `charts/supertrend_variant_trade_pnl_distribution.png` | Distribusi trade PnL perbandingan varian ST5_50 |
| `charts/supertrend_variant_march_2026_equity.png` | Equity khusus March 2026 perbandingan varian ST5_50 |
| `charts/short_reversal_switch_equity_curve.png` | Equity curve varian short-switch-to-long |
| `charts/short_reversal_switch_drawdown_curve.png` | Drawdown curve varian short-switch-to-long |
| `charts/short_reversal_switch_last30_equity.png` | Last 30D equity varian short-switch-to-long |
| `charts/short_switch_tp2r_p0_equity_curve.png` | Equity comparison baseline vs short-switch TP2R lama vs best P0 |
| `charts/short_switch_tp2r_p0_drawdown_curve.png` | Drawdown comparison baseline vs short-switch TP2R lama vs best P0 |
| `charts/short_switch_tp2r_p0_monthly_pnl_2026.png` | Monthly PnL 2026 comparison baseline vs short-switch TP2R lama vs best P0 |
| `charts/short_switch_tp2r_p0_rolling_windows.png` | Rolling PnL/DD comparison baseline vs short-switch TP2R lama vs best P0 |
| `charts/short_switch_tp2r_p0_last30_equity.png` | Last 30D equity comparison baseline vs short-switch TP2R lama vs best P0 |
| `monte_carlo/monte_pnl_fan_30d.png` | Baseline-control Monte Carlo fan chart 30D |
| `monte_carlo/monte_final_pnl_cdf_30d.png` | Baseline-control Monte Carlo final PnL CDF 30D |
| `monte_carlo/monte_maxdd_hist_30d.png` | Baseline-control Monte Carlo MaxDD histogram 30D |
| `monte_carlo/monte_pnl_fan_100d.png` | Baseline-control Monte Carlo fan chart 100D |
| `supertrend_regime_audit.md` | Audit grid SuperTrend 5m/15m ATR 5/10/20/50 |
| `supertrend_filter_candidates.csv` | Semua kandidat kombinasi bullish SuperTrend |
| `supertrend_variant_comparison.md` | Perbandingan baseline, ST5_50, long+short, dan long+short ST aligned |
| `supertrend_variant_comparison.csv` | Tabel machine-readable untuk perbandingan variant ST5_50 |
| `short_reversal_switch_comparison.md` | Audit short breakout yang switch ke long saat OR high reclaim |
| `short_reversal_switch_comparison.csv` | Summary varian short TP 1R/1.5R/2R |
| `short_reversal_switch_events.csv` | Sequence-level event varian short-switch |
| `short_reversal_switch_legs.csv` | Leg-level attribution varian short-switch |
| `short_switch_tp2r_p0_sweep.md` | P0 sweep short-switch TP2R dengan ST filter, asymmetric risk, dan switch buffer |
| `short_switch_tp2r_p0_full_report.md` | Full report kandidat P0 terbaik: yearly, YTD, month-to-month |
| `short_switch_tp2r_p0_sweep.csv` | Summary machine-readable P0 short-switch TP2R |
| `short_switch_tp2r_p0_best_events.csv` | Sequence-level event kandidat P0 terbaik |
| `short_switch_tp2r_p0_best_legs.csv` | Leg-level attribution kandidat P0 terbaik |
| `short_switch_tp2r_p0_best_yearly.csv` | Yearly metrics kandidat P0 terbaik |
| `short_switch_tp2r_p0_best_monthly.csv` | Monthly metrics kandidat P0 terbaik |

### 17.2 Canonical Data

```text
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/events.parquet
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/summary.json
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/manifest.json
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/package_gate.json
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/supertrend_regime_features.parquet
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/supertrend_regime_manifest.json
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/supertrend_variant_comparison_manifest.json
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/short_reversal_switch_comparison_manifest.json
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/short_switch_tp2r_p0_sweep_manifest.json
data/Level_0_Raw/MNQ_1m.duckdb
data/Level_0_Raw/MNQ_1m_duckdb_manifest.json
data/Level_0_Raw/MNQ_1m_yfinance_append_manifest.json
data/Level_0_Raw/MNQ_1m_continuity_report.json
data/Level_0_Raw/MNQ_yfinance_timeframe_parity_report.json
data/Level_1_Features/mnq/ORB/context.parquet
data/Level_1_Features/mnq/ORB/context_manifest.json
data/Level_1_Features/mnq/ORB/l1_audit.json
data/Level_1_Features/mnq/ORB/daily_confluence.parquet
data/Level_1_Features/mnq/ORB/daily_confluence_manifest.json
data/Level_1_Features/mnq/ORB/daily_confluence_audit.json
data/Level_2_Datamart/mnq/ORB/sweeps/sweep_events.parquet
data/Level_2_Datamart/mnq/ORB/sweeps/sweep_manifest.json
```

---

## 18. Lampiran A - 10 Trade Terakhir

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

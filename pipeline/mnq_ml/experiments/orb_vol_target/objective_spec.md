# Blueprint Strategi Algoritma: MNQ Opening Range Breakout Volatility Targeted

## 1. Filosofi Inti

Strategi ini mengkapitalisasi ketidakseimbangan momentum pada awal pembukaan
sesi New York. Hipotesisnya: breakout valid di atas opening range awal dapat
menangkap continuation intraday pada instrumen trend-following seperti Nasdaq
futures.

Prinsip desain:

- Price action first.
- Long only adalah jalur utama sementara karena hasil short/full long_short
  kurang bersih secara full-history.
- Tidak menambahkan filter indikator di baseline untuk menghindari curve-fitting.
- Position sizing berbasis fixed dollar risk.

## 2. Instrumen dan Sesi

- Instrumen: MNQ / Micro Nasdaq futures.
- Timezone keputusan: America/New_York.
- Timeframe keputusan: M1.
- Opening range candidates: 10m, 15m, 20m, 30m setelah 09:30 NY.
- Time exit: 15:00 NY.

## 3. Execution Rules

Opening range:

- Ambil high dan low dari N bar M1 pertama setelah 09:30 NY.
- N saat ini disweep di `10, 15, 20, 30`.
- Hari hanya eligible jika semua bar OR lengkap dan tidak mengandung source gap.

Entry:

- Long only.
- Long trigger saat candle M1 setelah opening range close di atas OR high.
- Short trigger dapat disweep, tetapi bukan jalur utama saat ini.
- Maksimal satu posisi per NY trading day.
- Signal memakai close bar t.
- Entry dieksekusi pada open bar t+1.

Exit:

- Mode baseline: tutup pada close bar 15:00 NY.
- Mode sweep terbaik saat ini: jika sudah mencapai +2R sebelum 15:00 NY,
  tutup dulu; jika tidak, tutup pada 15:00 NY.
- OR low dipakai sebagai referensi risk untuk position sizing, bukan sebagai
  intraday stop-exit pada baseline long.

## 4. Volatility Targeting

Target risk tetap dalam USD.

```text
risk_per_contract_usd = (entry_price - stop_price) * point_value_usd
contracts_float = target_risk_usd / risk_per_contract_usd
contracts_floor = floor(contracts_float)
contracts_used = min(max_contracts, contracts_floor)
```

Untuk futures live, kontrak harus integer. Baseline datamart menyimpan
`contracts_float`, `contracts_floor`, dan `contracts_used`. Trade dengan
`contracts_used < 1` ditolak karena tidak bisa dieksekusi tanpa melampaui target
risk.

## 5. Baseline Objective

Primary objective:

```text
maximize net expectancy after slippage and commission
```

Primary labels:

```text
label = 1 if net pnl per contract > 0
label = 0 otherwise
```

Evaluation fields:

- `pnl_per_contract_usd`
- `pnl_vol_target_usd`
- `r_multiple`
- `exit_reason`
- `contracts_used`
- `orb_range_pts`
- `entry_risk_pts`

## 6. Current Evidence

Latest sweep:

```text
orb_minutes: 10, 15, 20, 30
side_mode: long, short, long_short
target_risk_usd: 100, 200, 300, 400, 500, 600
exit_mode: time_exit, tp_2r_or_time
```

Best current candidate:

```text
15m OR, long only, TP 2R or 15:00 NY, risk $500
```

Observed short-window performance:

| Window | Trades | PnL | Max DD |
| ---: | ---: | ---: | ---: |
| 30D | 18 | $3,491 | -$549 |
| 50D | 30 | $5,448 | -$859 |
| 100D | 54 | $4,135 | -$4,066 |
| 200D | 94 | $5,569 | -$4,556 |

## 7. No-Lookahead Contract

- OR levels use only bars ending inside the selected opening range.
- Signal uses only bar t close.
- Entry uses next M1 bar open.
- Outcome exits at either TP 2R hit after entry or the configured time-exit bar.
- No future/session outcome fields may be used as model features.

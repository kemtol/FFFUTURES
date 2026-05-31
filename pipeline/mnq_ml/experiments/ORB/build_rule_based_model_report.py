from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod"
MODEL_DIR = ROOT / "model/MNQ/ORB/rule_based_15m_long_tp2r_eod"
CHART_DIR = MODEL_DIR / "charts"
MONTE_DIR = MODEL_DIR / "monte_carlo"
RAW_BASE = (
    "https://raw.githubusercontent.com/kemtol/FFFUTURES/main/"
    "model/MNQ/ORB/rule_based_15m_long_tp2r_eod"
)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def raw(path: str) -> str:
    return f"{RAW_BASE}/{path}"


def usd(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def load_inputs() -> tuple[pd.DataFrame, dict]:
    events = pd.read_parquet(DATA_DIR / "events.parquet")
    events["ny_date"] = pd.to_datetime(events["ny_date"])
    events = events.sort_values(["ny_date", "entry_ts"]).reset_index(drop=True)

    with (DATA_DIR / "summary.json").open() as f:
        summary = json.load(f)

    return events, summary


def style_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def clean_svg(path: Path) -> None:
    text = path.read_text()
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n")


def save_fig(fig, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    svg_path = directory / f"{stem}.svg"
    png_path = directory / f"{stem}.png"
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=150)
    clean_svg(svg_path)


def save_equity_curve(events: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    equity = events["pnl_net_usd"].cumsum()
    ax.plot(events["ny_date"], equity, color="#0f766e", linewidth=1.8)
    ax.axhline(0, color="#334155", linewidth=0.8, alpha=0.5)
    style_axis(ax, "MNQ ORB Rule-Based Equity Curve", "Cumulative PnL ($)")
    fig.autofmt_xdate()
    fig.tight_layout()
    save_fig(fig, CHART_DIR, "equity_curve")
    plt.close(fig)


def save_drawdown_curve(events: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    equity = events["pnl_net_usd"].cumsum()
    drawdown = equity - equity.cummax()
    ax.fill_between(events["ny_date"], drawdown, 0, color="#dc2626", alpha=0.28)
    ax.plot(events["ny_date"], drawdown, color="#991b1b", linewidth=1.2)
    style_axis(ax, "MNQ ORB Rule-Based Drawdown", "Drawdown ($)")
    fig.autofmt_xdate()
    fig.tight_layout()
    save_fig(fig, CHART_DIR, "drawdown_curve")
    plt.close(fig)


def save_monthly_pnl(events: pd.DataFrame) -> None:
    monthly = events.assign(month=events["ny_date"].dt.to_period("M").astype(str))
    monthly = monthly.groupby("month", as_index=False)["pnl_net_usd"].sum()

    fig, ax = plt.subplots(figsize=(12, 4.8))
    colors = ["#0f766e" if x >= 0 else "#dc2626" for x in monthly["pnl_net_usd"]]
    ax.bar(monthly["month"], monthly["pnl_net_usd"], color=colors, width=0.85)
    ax.axhline(0, color="#334155", linewidth=0.8)
    style_axis(ax, "Monthly Net PnL", "PnL ($)")
    tick_step = max(1, len(monthly) // 14)
    ax.set_xticks(range(0, len(monthly), tick_step))
    ax.set_xticklabels(monthly["month"].iloc[::tick_step], rotation=45, ha="right")
    fig.tight_layout()
    save_fig(fig, CHART_DIR, "monthly_pnl")
    plt.close(fig)


def save_rolling_windows(summary: dict) -> None:
    order = ["5D", "10D", "20D", "30D", "50D", "100D", "200D"]
    rows = [
        {
            "window": window,
            "pnl": summary["windows"][window]["pnl_usd"],
            "dd": summary["windows"][window]["max_dd_usd"],
        }
        for window in order
    ]
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = range(len(df))
    ax.bar([i - 0.18 for i in x], df["pnl"], width=0.36, color="#0f766e", label="PnL")
    ax.bar([i + 0.18 for i in x], df["dd"], width=0.36, color="#dc2626", label="Max DD")
    ax.axhline(0, color="#334155", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["window"])
    ax.legend(frameon=False)
    style_axis(ax, "Recent Rolling Window PnL / DD", "$")
    fig.tight_layout()
    save_fig(fig, CHART_DIR, "rolling_windows")
    plt.close(fig)


def save_trade_pnl_distribution(events: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.hist(events["pnl_net_usd"], bins=45, color="#2563eb", alpha=0.78)
    ax.axvline(0, color="#334155", linewidth=0.9)
    style_axis(ax, "Trade PnL Distribution", "Trade count")
    ax.set_xlabel("Net PnL per trade ($)")
    fig.tight_layout()
    save_fig(fig, CHART_DIR, "trade_pnl_distribution")
    plt.close(fig)


def build_daily_pnl(events: pd.DataFrame) -> pd.Series:
    daily = events.groupby(events["ny_date"].dt.normalize())["pnl_net_usd"].sum().sort_index()
    idx = pd.date_range(daily.index.min(), daily.index.max(), freq="B")
    return daily.reindex(idx, fill_value=0.0)


def monte_carlo(daily_pnl: pd.Series, n_paths: int = 5000, seed: int = 260531) -> dict:
    rng = np.random.default_rng(seed)
    values = daily_pnl.to_numpy(dtype=float)
    horizons = [30, 100, 200]
    result: dict[str, dict] = {}

    for horizon in horizons:
        samples = rng.choice(values, size=(n_paths, horizon), replace=True)
        cumulative = samples.cumsum(axis=1)
        with_zero = np.concatenate([np.zeros((n_paths, 1)), cumulative], axis=1)
        peaks = np.maximum.accumulate(with_zero, axis=1)[:, 1:]
        drawdowns = cumulative - peaks
        final_pnl = cumulative[:, -1]
        max_dd = drawdowns.min(axis=1)

        result[f"{horizon}D"] = {
            "horizon": horizon,
            "paths": n_paths,
            "median_pnl_usd": float(np.median(final_pnl)),
            "p5_pnl_usd": float(np.percentile(final_pnl, 5)),
            "p95_pnl_usd": float(np.percentile(final_pnl, 95)),
            "prob_final_loss": float((final_pnl < 0).mean()),
            "median_max_dd_usd": float(np.median(max_dd)),
            "p5_max_dd_usd": float(np.percentile(max_dd, 5)),
            "prob_dd_breach_2000": float((max_dd <= -2000).mean()),
            "prob_hit_3000": float((cumulative.max(axis=1) >= 3000).mean()),
            "final_pnl": final_pnl,
            "max_dd": max_dd,
            "sample_paths": cumulative[:250],
        }

    return result


def save_monte_carlo_charts(mc: dict) -> None:
    for key in ["30D", "100D"]:
        horizon = mc[key]["horizon"]
        paths = mc[key]["sample_paths"]
        days = np.arange(1, horizon + 1)

        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        ax.plot(days, paths.T, color="#64748b", alpha=0.05, linewidth=0.8)
        ax.plot(days, np.median(paths, axis=0), color="#0f766e", linewidth=2.0, label="Median")
        ax.axhline(0, color="#334155", linewidth=0.8)
        ax.axhline(3000, color="#2563eb", linewidth=0.9, linestyle="--", label="+$3,000")
        style_axis(ax, f"Monte Carlo PnL Fan {key}", "Cumulative PnL ($)")
        ax.set_xlabel("Trading days")
        ax.legend(frameon=False)
        fig.tight_layout()
        save_fig(fig, MONTE_DIR, f"monte_pnl_fan_{key.lower()}")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    final_pnl = np.sort(mc["30D"]["final_pnl"])
    cdf = np.arange(1, len(final_pnl) + 1) / len(final_pnl)
    ax.plot(final_pnl, cdf, color="#2563eb", linewidth=1.8)
    ax.axvline(0, color="#334155", linewidth=0.9)
    ax.axvline(3000, color="#0f766e", linewidth=0.9, linestyle="--")
    style_axis(ax, "Monte Carlo Final PnL CDF 30D", "Cumulative probability")
    ax.set_xlabel("Final PnL ($)")
    fig.tight_layout()
    save_fig(fig, MONTE_DIR, "monte_final_pnl_cdf_30d")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.hist(mc["30D"]["max_dd"], bins=55, color="#dc2626", alpha=0.78)
    ax.axvline(-2000, color="#334155", linewidth=1.0, linestyle="--", label="-$2,000")
    style_axis(ax, "Monte Carlo Max Drawdown 30D", "Path count")
    ax.set_xlabel("Max drawdown ($)")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, MONTE_DIR, "monte_maxdd_hist_30d")
    plt.close(fig)


def monte_rows(mc: dict) -> str:
    rows = []
    for key in ["30D", "100D", "200D"]:
        item = mc[key]
        rows.append(
            "| {key} | {median} | {p5} | {loss} | {dd} | {mll} | {target} |".format(
                key=key,
                median=usd(item["median_pnl_usd"]),
                p5=usd(item["p5_pnl_usd"]),
                loss=pct(item["prob_final_loss"]),
                dd=usd(item["median_max_dd_usd"]),
                mll=pct(item["prob_dd_breach_2000"]),
                target=pct(item["prob_hit_3000"]),
            )
        )
    return "\n".join(rows)


def last_trades_rows(events: pd.DataFrame) -> str:
    rows = []
    for _, row in events.tail(10).iterrows():
        rows.append(
            "| {date} | {signal} | {exit_reason} | {contracts} | {pnl} |".format(
                date=row["ny_date"].strftime("%Y-%m-%d"),
                signal=str(row["signal_ts"])[:16],
                exit_reason=row["exit_reason"],
                contracts=int(row["contracts_used"]),
                pnl=usd(float(row["pnl_net_usd"])),
            )
        )
    return "\n".join(rows)


def build_report(events: pd.DataFrame, summary: dict, mc: dict) -> str:
    perf = summary["performance"]
    quality = summary["daily_quality"]
    costs = summary["costs"]
    signal = summary["signal_range"]
    windows = summary["windows"]

    window_order = ["5D", "10D", "20D", "30D", "50D", "100D", "200D"]
    window_rows = "\n".join(
        "| {window} | {trades:,} | {wr} | {pnl} | {dd} |".format(
            window=window,
            trades=int(windows[window]["trades"]),
            wr=pct(windows[window]["win_rate"]),
            pnl=usd(windows[window]["pnl_usd"]),
            dd=usd(windows[window]["max_dd_usd"]),
        )
        for window in window_order
    )

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return f"""# Strategi MNQ Opening Range Breakout Rule-Based Iterasi v1
**Evaluasi Baseline 15m Long TP2R/EOD**

Tanggal laporan: **{created_at[:10]}**

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
| Periode sinyal | {signal["min_signal_ts"][:10]} - {signal["max_signal_ts"][:10]} |
| Total trade | {int(perf["trades"]):,} |
| Win rate | {pct(perf["win_rate"])} |
| Net PnL | {usd(perf["total_pnl_usd"])} |
| Max drawdown | {usd(perf["max_dd_usd"])} |
| Profit factor | {perf["profit_factor"]:.2f} |
| Daily Sharpe | {quality["daily_sharpe_annualized"]:.2f} |
| Daily Sortino | {quality["daily_sortino_annualized"]:.2f} |
| 30D terakhir | {int(windows["30D"]["trades"])} trade, {usd(windows["30D"]["pnl_usd"])} PnL, {usd(windows["30D"]["max_dd_usd"])} max DD |

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

![Equity Curve]({raw("charts/equity_curve.png")})

Equity curve menunjukkan strategi menghasilkan PnL positif secara historis,
tetapi jalurnya tidak linear. Ada fase panjang yang relatif datar dan beberapa
periode drawdown besar.

### 5.2 Drawdown

![Drawdown Curve]({raw("charts/drawdown_curve.png")})

Drawdown maksimum historis sebesar {usd(perf["max_dd_usd"])}. Ini jauh lebih
besar daripada batas MLL Topstep 50K, sehingga evaluasi live tidak boleh hanya
mengandalkan total PnL historis.

### 5.3 Monthly PnL

![Monthly PnL]({raw("charts/monthly_pnl.png")})

Grafik bulanan membantu melihat bahwa strategi tidak menghasilkan distribusi
profit yang stabil setiap bulan. Ada bulan kuat, bulan kosong, dan bulan rugi.

### 5.4 Distribusi PnL Per Trade

![Trade PnL Distribution]({raw("charts/trade_pnl_distribution.png")})

Rata-rata loss per trade masih lebih besar daripada rata-rata win. Edge muncul
dari kombinasi win rate 56.48%, sizing, dan beberapa periode momentum yang
produktif.

---

## 6. Metrik Historis

| Metric | Value |
| --- | ---: |
| Signal range | {signal["min_signal_ts"][:10]} to {signal["max_signal_ts"][:10]} |
| Trades | {int(perf["trades"]):,} |
| Win rate | {pct(perf["win_rate"])} |
| Net PnL | {usd(perf["total_pnl_usd"])} |
| Gross profit | {usd(perf["gross_profit_usd"])} |
| Gross loss | {usd(perf["gross_loss_usd"])} |
| Profit factor | {perf["profit_factor"]:.2f} |
| Max drawdown | {usd(perf["max_dd_usd"])} |
| Return / DD | {perf["return_dd"]:.2f} |
| Expectancy / trade | {usd(perf["expectancy_per_trade_usd"])} |
| Median trade | {usd(perf["median_pnl_usd"])} |
| Average win | {usd(perf["avg_win_usd"])} |
| Average loss | {usd(perf["avg_loss_usd"])} |
| Payoff ratio | {perf["payoff_ratio"]:.2f} |
| Average contracts | {perf["avg_contracts"]:.2f} |
| Max consecutive wins | {int(perf["max_consecutive_wins"])} |
| Max consecutive losses | {int(perf["max_consecutive_losses"])} |

---

## 7. Cost Model

| Cost | Value |
| --- | ---: |
| Commission + fees | ${costs["commission_round_turn_usd_per_contract"]:.2f} RT / contract |
| Slippage | {costs["slippage_ticks_per_side"]} tick per side |
| Modeled slippage | ${costs["slippage_round_turn_usd_per_contract"]:.2f} RT / contract |
| Total commission paid | {usd(costs["total_commission_paid_usd"])} |
| Total modeled slippage | {usd(costs["total_modeled_slippage_usd"])} |

Biaya sudah dimasukkan pada `pnl_net_usd`: TopstepX MNQ $1.24 round-turn per
contract dan modeled slippage 1 tick per side.

---

## 8. Daily Quality

Sharpe and Sortino are computed from daily dollar PnL over MNQ NY session days,
with zero PnL on no-trade days, annualized by `sqrt(252)`.

| Metric | Value |
| --- | ---: |
| Trading days measured | {int(quality["trading_days"]):,} |
| Active days | {int(quality["active_days"]):,} |
| Active-day rate | {pct(quality["active_day_rate"])} |
| Active-day win rate | {pct(quality["active_day_win_rate"])} |
| Daily average PnL | {usd(quality["daily_avg_pnl_usd"])} |
| Daily PnL std dev | {usd(quality["daily_std_pnl_usd"])} |
| Daily Sharpe | {quality["daily_sharpe_annualized"]:.2f} |
| Daily Sortino | {quality["daily_sortino_annualized"]:.2f} |
| Best day | {usd(quality["best_day_pnl_usd"])} |
| Worst day | {usd(quality["worst_day_pnl_usd"])} |
| Best-day profit share | {pct(quality["best_day_profit_share"])} |
| 50% consistency flag | {"Pass" if quality["topstep_50pct_consistency_ok"] else "Fail"} |

---

## 9. Rolling Window Terakhir

![Rolling Windows]({raw("charts/rolling_windows.png")})

| Window | Trades | Win Rate | PnL | Max DD |
| ---: | ---: | ---: | ---: | ---: |
{window_rows}

Interpretasi:

- 30D terakhir adalah bagian paling menarik: 18 trade dan {usd(windows["30D"]["pnl_usd"])} PnL.
- 5D dan 10D masih terlalu pendek untuk menjadi bukti edge.
- 100D dan 200D tetap positif, tetapi DD historisnya mulai berat untuk Topstep.

---

## 10. Monte Carlo dan Stress Test

Monte Carlo dilakukan dengan bootstrap dari daily PnL historis. Ini bukan
prediksi masa depan, tetapi stress test distribusi jika pola daily PnL historis
muncul dalam urutan yang berbeda.

| Horizon | Median PnL | P5 PnL | Prob. Akhir Rugi | Median MaxDD | Prob. DD <= -$2k | Prob. Hit +$3k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{monte_rows(mc)}

### 10.1 Fan Chart 30D

![Monte Carlo PnL Fan 30D]({raw("monte_carlo/monte_pnl_fan_30d.png")})

### 10.2 Distribusi Final PnL 30D

![Monte Carlo Final PnL CDF 30D]({raw("monte_carlo/monte_final_pnl_cdf_30d.png")})

### 10.3 Max Drawdown 30D

![Monte Carlo MaxDD 30D]({raw("monte_carlo/monte_maxdd_hist_30d.png")})

### 10.4 Fan Chart 100D

![Monte Carlo PnL Fan 100D]({raw("monte_carlo/monte_pnl_fan_100d.png")})

Kesimpulan Monte Carlo: strategi punya upside untuk mencapai +$3,000 dalam
sebagian path 30D, tetapi risiko drawdown terhadap batas -$2,000 tetap perlu
diuji lebih ketat dengan simulator Topstep yang memperhitungkan aturan akun.

---

## 11. Penilaian Risiko

### 11.1 Risiko Drawdown

Max drawdown historis {usd(perf["max_dd_usd"])} jauh lebih besar daripada MLL
Topstep 50K. Ini tidak otomatis membatalkan strategi, karena evaluasi Topstep
berjalan pada window pendek, tetapi artinya strategi membutuhkan guard dan
monitoring harian.

### 11.2 Risiko No Normal SL

Strategi ini tidak memakai SL normal. Exit loss terjadi lewat time exit.
Konsekuensinya, flash drop atau trend day yang berlawanan bisa menghasilkan
kerugian lebih besar dari target risk teoritis. Catastrophic guard harus
dipilih sebagai layer operasional terpisah.

### 11.3 Risiko Curve Fit

Baseline ini cukup bersih karena hanya memakai OR 15m, long only, TP 2R/time
exit, dan risk $500. Namun pemilihan parameter tetap berasal dari sweep, jadi
forward test diperlukan sebelum dianggap valid.

### 11.4 Risiko Eksekusi Live

Live version harus memastikan:

- M1 candle close sudah final sebelum entry.
- Entry dilakukan pada open M1 berikutnya.
- Jam New York dan daylight saving benar.
- Tidak ada duplicate trade per hari.
- Tidak ada posisi tanpa catastrophic guard.
- Data feed dan broker connection punya heartbeat.

---

## 12. Rekomendasi Sementara

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

## 13. Keputusan Sementara

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

## 14. Artifact Register

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
| `monte_carlo/monte_pnl_fan_30d.png` | Monte Carlo fan chart 30D |
| `monte_carlo/monte_final_pnl_cdf_30d.png` | Monte Carlo final PnL CDF 30D |
| `monte_carlo/monte_maxdd_hist_30d.png` | Monte Carlo MaxDD histogram 30D |
| `monte_carlo/monte_pnl_fan_100d.png` | Monte Carlo fan chart 100D |

### Canonical Data

```text
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/events.parquet
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/summary.json
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/manifest.json
```

---

## 15. Lampiran A - 10 Trade Terakhir

| NY Date | Signal UTC | Exit | Contracts | Net PnL |
| --- | --- | --- | ---: | ---: |
{last_trades_rows(events)}
"""


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    MONTE_DIR.mkdir(parents=True, exist_ok=True)

    events, summary = load_inputs()
    daily_pnl = build_daily_pnl(events)
    mc = monte_carlo(daily_pnl)

    save_equity_curve(events)
    save_drawdown_curve(events)
    save_monthly_pnl(events)
    save_rolling_windows(summary)
    save_trade_pnl_distribution(events)
    save_monte_carlo_charts(mc)

    (MODEL_DIR / "REPORT.md").write_text(build_report(events, summary, mc))
    (MODEL_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    serializable_mc = {
        key: {k: v for k, v in value.items() if k not in {"final_pnl", "max_dd", "sample_paths"}}
        for key, value in mc.items()
    }
    (MODEL_DIR / "monte_carlo_metrics.json").write_text(
        json.dumps(serializable_mc, indent=2, sort_keys=True) + "\n"
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy_id": summary["strategy_id"],
        "source_events": rel(DATA_DIR / "events.parquet"),
        "source_summary": rel(DATA_DIR / "summary.json"),
        "outputs": {
            "report": rel(MODEL_DIR / "REPORT.md"),
            "metrics": rel(MODEL_DIR / "metrics.json"),
            "charts": [
                rel(CHART_DIR / "equity_curve.svg"),
                rel(CHART_DIR / "equity_curve.png"),
                rel(CHART_DIR / "drawdown_curve.svg"),
                rel(CHART_DIR / "drawdown_curve.png"),
                rel(CHART_DIR / "monthly_pnl.svg"),
                rel(CHART_DIR / "monthly_pnl.png"),
                rel(CHART_DIR / "rolling_windows.svg"),
                rel(CHART_DIR / "rolling_windows.png"),
                rel(CHART_DIR / "trade_pnl_distribution.svg"),
                rel(CHART_DIR / "trade_pnl_distribution.png"),
                rel(MONTE_DIR / "monte_pnl_fan_30d.svg"),
                rel(MONTE_DIR / "monte_pnl_fan_30d.png"),
                rel(MONTE_DIR / "monte_final_pnl_cdf_30d.svg"),
                rel(MONTE_DIR / "monte_final_pnl_cdf_30d.png"),
                rel(MONTE_DIR / "monte_maxdd_hist_30d.svg"),
                rel(MONTE_DIR / "monte_maxdd_hist_30d.png"),
                rel(MONTE_DIR / "monte_pnl_fan_100d.svg"),
                rel(MONTE_DIR / "monte_pnl_fan_100d.png"),
            ],
            "monte_carlo_metrics": rel(MODEL_DIR / "monte_carlo_metrics.json"),
        },
    }
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

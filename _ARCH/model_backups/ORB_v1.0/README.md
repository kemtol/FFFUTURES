# ORB Reversal Model v1.0

LightGBM binary classifier untuk filter reversal ORB setups pada MGC (Micro Gold Futures).

## Quickstart

```python
import lightgbm as lgb
import pandas as pd

model = lgb.Booster(model_file="lgbm_rev_1r2_120m.txt")

# Features required (in order):
# ['orb_range_atr_ratio', 'breakout_strength', 'atr14_at_entry', 'price_vs_vwap_pct', 'adx_14_15m', 'ema_slope_1h', 'day_of_week', 'time_in_session_min', 'orb_tf', 'session', 'breakout_side']

prob = model.predict(X)  # probability TP 2R hit within 120m
signal = prob >= 0.50    # threshold — lihat REPORT.md untuk kalibrasi
```

## Files

| File | Deskripsi |
|------|-----------|
| `lgbm_rev_1r2_120m.txt` | Model LightGBM |
| `lgbm_rev_1r2_120m_meta.json` | Metadata, CV results, feature importance |
| `REPORT.md` | Laporan lengkap training run ini |

## Key Numbers

- CV Mean AUC: **0.6234**
- CV Mean exp net (top 40%): **0.485R**
- Training data: 2010–2023, 28,884 reversal events
- Holdout: Jan 2024+ (belum dievaluasi)

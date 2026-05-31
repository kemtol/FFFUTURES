# MNQ ORB Volatility Target Level 1

Level 1 context for the `orb_vol_target` experiment.

## Files

| File | Rows | Columns | Grain | Description |
| --- | ---: | ---: | --- | --- |
| `context.parquet` | 2,487,265 | 18 | one row per M1 bar | New York session ORB context |
| `context_manifest.json` | - | - | manifest | Build summary |
| `l1_audit.json` | - | - | audit report | L0-to-L1 authenticity, integrity, continuity, completeness |
| `daily_confluence.parquet` | 2,199 | 31 | one row per MNQ NY date | Prior daily SPY/QQQ/VIX/TNX/DXY confluence |
| `daily_confluence_manifest.json` | - | - | manifest | Daily confluence feature list and source summary |
| `daily_confluence_audit.json` | - | - | audit report | Daily confluence null, source-date, and no-lookahead audit |

## `context.parquet`

Columns:

```text
timestamp_utc
open
high
low
close
volume
source_bar_count
contains_source_gap
ny_date
ny_time
minutes_from_open
bar_data_quality_ok
orb_high
orb_low
orb_bar_count
orb_range_pts
orb_complete
eligible_after_or
```

Timing contract:

```text
Source timeframe: M1
Canonical baseline opening range: 09:30-10:00 New York time
Sweep opening ranges: 10m, 15m, 20m, 30m after 09:30 NY
Canonical ORB bars: right-labeled M1 bars ending 09:31 through 10:00
Sweep worker recomputes OR levels from raw M1 context per OR duration
```

Quality contract:

```text
bar_data_quality_ok = source_bar_count == 1 and contains_source_gap == false
orb_complete = all 30 ORB bars are quality-valid for the NY date
eligible_after_or = quality-valid post-ORB bar on a complete ORB day
```

No-lookahead note:

`orb_high`, `orb_low`, and `orb_range_pts` are known only after the 10:00 NY
opening range completes. They may be used for post-10:00 decision rows, but not
for pre-10:00 decisions.

For non-30m OR durations, use the sweep worker rather than these canonical
30m columns:

```text
pipeline/mnq_ml/experiments/orb_vol_target/sweep_orb_params.py
```

## `daily_confluence.parquet`

Columns:

```text
ny_date
daily_confluence_feature_date
29 dc_* feature columns
```

Feature families:

| Prefix | Description |
| --- | --- |
| `dc_spy_*` | Broad-market daily trend, stretch, and realized volatility |
| `dc_qqq_*` | Nasdaq daily trend, stretch, realized volatility, and SPY-relative strength |
| `dc_vix_*` | Volatility level, change, percentile, and stretch |
| `dc_tnx_*` | 10Y yield level and change |
| `dc_dxy_*` | US dollar short-term trend |

No-lookahead note:

For MNQ trade date `D`, daily confluence must use only external daily rows
strictly earlier than `D`. The audit enforces
`daily_confluence_feature_date < ny_date`.

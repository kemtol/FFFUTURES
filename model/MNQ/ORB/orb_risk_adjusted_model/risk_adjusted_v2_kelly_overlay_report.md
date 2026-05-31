# MNQ ORB Base-Floor Kelly Overlay V2

Created: `2026-05-30T10:11:28.050555+00:00`

## Contract

- Base risk: `$500`
- Payoff ratio `b`: `2.0`
- Break-even probability: `0.3333`
- Min/max risk multiplier: `1.0` / `2.0`
- Kelly fractions: `[0.1, 0.25, 0.5, 1.0]`
- Normalized target risks: `[600.0, 750.0, 1000.0]`

Base-floor fractional Kelly formula:

```text
risk_multiplier = clip(1 + kelly_fraction * max(0, (b*p - (1-p))/b), min=1.0, max)
```

Base-floor normalized Kelly formula:

```text
scale is fit on train only; risk_multiplier = clip(1 + scale * max(0, (b*p - (1-p))/b), min=1.0, max)
```

Continuous risk sizing: `pnl_usd = r_multiple * risk_usd`.

Executable integer sizing:

```text
contracts_minrisk_ceil = max(1, ceil(risk_usd / risk_per_contract_usd))
integer_pnl_usd = r_multiple * contracts_minrisk_ceil * risk_per_contract_usd
```

## Train

| Variant | Rows | Cont PnL | Cont DD | Cont R/DD | Int PnL | Int DD | Int R/DD | Avg desired risk | Avg int risk | Under 1ct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `basefloor_kelly_0.10x` | 2068 | $40,686 | $-13,236 | 3.07 | $45,978 | $-15,454 | 2.98 | $511 | $588 | 0.2% |
| `basefloor_kelly_0.25x` | 2068 | $44,282 | $-13,779 | 3.21 | $50,851 | $-16,420 | 3.10 | $529 | $605 | 0.2% |
| `basefloor_kelly_0.50x` | 2068 | $50,274 | $-14,683 | 3.42 | $52,848 | $-17,870 | 2.96 | $557 | $634 | 0.2% |
| `basefloor_kelly_1.00x` | 2068 | $62,259 | $-16,493 | 3.77 | $64,769 | $-18,772 | 3.45 | $614 | $692 | 0.2% |
| `fixed_1.00x` | 2068 | $38,289 | $-12,874 | 2.97 | $41,403 | $-15,092 | 2.74 | $500 | $577 | 0.2% |
| `norm_target_1000` | 2068 | $83,394 | $-22,025 | 3.79 | $85,636 | $-24,084 | 3.56 | $840 | $916 | 0.2% |
| `norm_target_600` | 2068 | $59,252 | $-16,039 | 3.69 | $61,698 | $-19,136 | 3.22 | $600 | $677 | 0.2% |
| `norm_target_750` | 2068 | $75,835 | $-21,358 | 3.55 | $75,912 | $-23,456 | 3.24 | $750 | $827 | 0.2% |

## Validation

| Variant | Rows | Cont PnL | Cont DD | Cont R/DD | Int PnL | Int DD | Int R/DD | Avg desired risk | Avg int risk | Under 1ct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `basefloor_kelly_0.10x` | 351 | $5,486 | $-7,824 | 0.70 | $7,570 | $-8,915 | 0.85 | $512 | $629 | 0.6% |
| `basefloor_kelly_0.25x` | 351 | $5,526 | $-8,140 | 0.68 | $5,407 | $-9,991 | 0.54 | $531 | $651 | 0.6% |
| `basefloor_kelly_0.50x` | 351 | $5,594 | $-8,667 | 0.65 | $8,518 | $-9,434 | 0.90 | $562 | $682 | 0.6% |
| `basefloor_kelly_1.00x` | 351 | $5,729 | $-9,871 | 0.58 | $6,498 | $-11,960 | 0.54 | $625 | $741 | 0.6% |
| `fixed_1.00x` | 351 | $5,459 | $-7,613 | 0.72 | $7,207 | $-9,136 | 0.79 | $500 | $615 | 0.6% |
| `norm_target_1000` | 351 | $2,356 | $-14,196 | 0.17 | $3,740 | $-14,682 | 0.25 | $850 | $967 | 0.6% |
| `norm_target_600` | 351 | $5,695 | $-9,505 | 0.60 | $6,883 | $-11,908 | 0.58 | $609 | $727 | 0.6% |
| `norm_target_750` | 351 | $2,732 | $-13,451 | 0.20 | $2,752 | $-15,336 | 0.18 | $763 | $881 | 0.6% |

## Holdout

| Variant | Rows | Cont PnL | Cont DD | Cont R/DD | Int PnL | Int DD | Int R/DD | Avg desired risk | Avg int risk | Under 1ct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `basefloor_kelly_0.10x` | 140 | $13,917 | $-3,242 | 4.29 | $18,076 | $-3,436 | 5.26 | $511 | $669 | 2.9% |
| `basefloor_kelly_0.25x` | 140 | $14,662 | $-3,432 | 4.27 | $17,827 | $-4,238 | 4.21 | $528 | $678 | 2.9% |
| `basefloor_kelly_0.50x` | 140 | $15,903 | $-3,749 | 4.24 | $18,523 | $-4,238 | 4.37 | $557 | $698 | 2.9% |
| `basefloor_kelly_1.00x` | 140 | $18,385 | $-4,383 | 4.19 | $22,006 | $-4,954 | 4.44 | $614 | $756 | 2.9% |
| `fixed_1.00x` | 140 | $13,420 | $-3,115 | 4.31 | $17,140 | $-3,430 | 5.00 | $500 | $652 | 2.9% |
| `norm_target_1000` | 140 | $23,915 | $-5,565 | 4.30 | $26,460 | $-6,258 | 4.23 | $818 | $969 | 2.9% |
| `norm_target_600` | 140 | $17,762 | $-4,224 | 4.21 | $21,676 | $-4,954 | 4.38 | $600 | $746 | 2.9% |
| `norm_target_750` | 140 | $23,900 | $-5,578 | 4.28 | $27,857 | $-6,220 | 4.48 | $732 | $883 | 2.9% |

## Readout

- Desired risk is floored at the base $500 risk; Kelly only adds risk above the baseline.
- Normalized variants fit their scaling on train only, then apply the same scale to validation/holdout.
- Max multiplier caps desired risk; default cap is 2.0x base risk.
- Integer MNQ execution is the practical constraint: executable contracts are rounded up so actual risk is at least the desired risk.

## Recent Windows

### `fixed_1.00x`

| Window | Trades | Int PnL | Int DD | Int R/DD | Avg Int Risk |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5D | 7 | $-243 | $-2,142 | -0.11 | $751 |
| 10D | 15 | $-887 | $-2,142 | -0.41 | $755 |
| 20D | 29 | $2,296 | $-2,142 | 1.07 | $710 |
| 30D | 45 | $1,423 | $-3,430 | 0.41 | $681 |
| 50D | 73 | $4,454 | $-3,430 | 1.30 | $654 |
| 100D | 137 | $18,404 | $-3,430 | 5.36 | $653 |
| 200D | 276 | $16,711 | $-10,940 | 1.53 | $638 |

### `basefloor_kelly_0.10x`

| Window | Trades | Int PnL | Int DD | Int R/DD | Avg Int Risk |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5D | 7 | $-243 | $-2,142 | -0.11 | $751 |
| 10D | 15 | $-887 | $-2,142 | -0.41 | $755 |
| 20D | 29 | $2,558 | $-2,142 | 1.19 | $714 |
| 30D | 45 | $1,680 | $-3,436 | 0.49 | $699 |
| 50D | 73 | $5,283 | $-3,436 | 1.54 | $671 |
| 100D | 137 | $19,340 | $-3,436 | 5.63 | $670 |
| 200D | 276 | $17,738 | $-10,718 | 1.65 | $653 |

### `basefloor_kelly_1.00x`

| Window | Trades | Int PnL | Int DD | Int R/DD | Avg Int Risk |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5D | 7 | $-243 | $-2,142 | -0.11 | $751 |
| 10D | 15 | $-2,009 | $-2,445 | -0.82 | $793 |
| 20D | 29 | $3,151 | $-2,445 | 1.29 | $763 |
| 30D | 45 | $1,794 | $-3,914 | 0.46 | $747 |
| 50D | 73 | $5,931 | $-4,954 | 1.20 | $758 |
| 100D | 137 | $23,270 | $-4,954 | 4.70 | $758 |
| 200D | 276 | $20,148 | $-13,648 | 1.48 | $748 |

### `norm_target_600`

| Window | Trades | Int PnL | Int DD | Int R/DD | Avg Int Risk |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5D | 7 | $-243 | $-2,142 | -0.11 | $751 |
| 10D | 15 | $-2,009 | $-2,445 | -0.82 | $793 |
| 20D | 29 | $2,870 | $-2,445 | 1.17 | $758 |
| 30D | 45 | $1,410 | $-4,017 | 0.35 | $739 |
| 50D | 73 | $5,712 | $-4,954 | 1.15 | $750 |
| 100D | 137 | $22,939 | $-4,954 | 4.63 | $749 |
| 200D | 276 | $19,530 | $-13,595 | 1.44 | $735 |

### `norm_target_750`

| Window | Trades | Int PnL | Int DD | Int R/DD | Avg Int Risk |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5D | 7 | $-526 | $-2,916 | -0.18 | $857 |
| 10D | 15 | $-2,970 | $-3,124 | -0.95 | $894 |
| 20D | 29 | $3,802 | $-3,124 | 1.22 | $871 |
| 30D | 45 | $2,778 | $-4,034 | 0.69 | $863 |
| 50D | 73 | $7,758 | $-6,220 | 1.25 | $904 |
| 100D | 137 | $28,263 | $-6,220 | 4.54 | $886 |
| 200D | 276 | $22,670 | $-16,792 | 1.35 | $884 |

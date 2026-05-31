# MNQ ORB Base-Floor Kelly Overlay V2

Created: `2026-05-29T05:40:35.564841+00:00`

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
| `basefloor_kelly_0.10x` | 2068 | $39,699 | $-13,497 | 2.94 | $44,430 | $-15,968 | 2.78 | $512 | $589 | 0.2% |
| `basefloor_kelly_0.25x` | 2068 | $41,814 | $-14,432 | 2.90 | $49,422 | $-17,488 | 2.83 | $529 | $605 | 0.2% |
| `basefloor_kelly_0.50x` | 2068 | $45,338 | $-15,990 | 2.84 | $44,831 | $-18,768 | 2.39 | $558 | $634 | 0.2% |
| `basefloor_kelly_1.00x` | 2068 | $52,387 | $-19,106 | 2.74 | $56,963 | $-21,658 | 2.63 | $616 | $693 | 0.2% |
| `fixed_1.00x` | 2068 | $38,289 | $-12,874 | 2.97 | $41,403 | $-15,092 | 2.74 | $500 | $577 | 0.2% |
| `norm_target_1000` | 2068 | $72,627 | $-22,363 | 3.25 | $74,974 | $-24,572 | 3.05 | $859 | $935 | 0.2% |
| `norm_target_600` | 2068 | $50,434 | $-18,243 | 2.76 | $52,414 | $-21,236 | 2.47 | $600 | $677 | 0.2% |
| `norm_target_750` | 2068 | $59,385 | $-27,331 | 2.17 | $60,286 | $-30,308 | 1.99 | $750 | $828 | 0.2% |

## Validation

| Variant | Rows | Cont PnL | Cont DD | Cont R/DD | Int PnL | Int DD | Int R/DD | Avg desired risk | Avg int risk | Under 1ct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `basefloor_kelly_0.10x` | 351 | $5,349 | $-7,790 | 0.69 | $7,023 | $-8,915 | 0.79 | $511 | $627 | 0.6% |
| `basefloor_kelly_0.25x` | 351 | $5,183 | $-8,055 | 0.64 | $4,894 | $-9,991 | 0.49 | $529 | $645 | 0.6% |
| `basefloor_kelly_0.50x` | 351 | $4,908 | $-8,496 | 0.58 | $6,451 | $-9,144 | 0.71 | $557 | $672 | 0.6% |
| `basefloor_kelly_1.00x` | 351 | $4,356 | $-9,831 | 0.44 | $3,930 | $-11,616 | 0.34 | $615 | $734 | 0.6% |
| `fixed_1.00x` | 351 | $5,459 | $-7,613 | 0.72 | $7,207 | $-9,136 | 0.79 | $500 | $615 | 0.6% |
| `norm_target_1000` | 351 | $1,958 | $-15,217 | 0.13 | $4,114 | $-15,346 | 0.27 | $843 | $963 | 0.6% |
| `norm_target_600` | 351 | $4,509 | $-9,432 | 0.48 | $4,409 | $-12,010 | 0.37 | $599 | $717 | 0.6% |
| `norm_target_750` | 351 | $1,938 | $-13,234 | 0.15 | $3,584 | $-15,124 | 0.24 | $740 | $856 | 0.6% |

## Holdout

| Variant | Rows | Cont PnL | Cont DD | Cont R/DD | Int PnL | Int DD | Int R/DD | Avg desired risk | Avg int risk | Under 1ct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `basefloor_kelly_0.10x` | 140 | $13,799 | $-3,236 | 4.26 | $17,838 | $-3,436 | 5.19 | $510 | $665 | 2.9% |
| `basefloor_kelly_0.25x` | 140 | $14,368 | $-3,418 | 4.20 | $18,518 | $-3,910 | 4.74 | $524 | $679 | 2.9% |
| `basefloor_kelly_0.50x` | 140 | $15,315 | $-3,722 | 4.11 | $17,863 | $-4,238 | 4.22 | $548 | $693 | 2.9% |
| `basefloor_kelly_1.00x` | 140 | $17,209 | $-4,329 | 3.98 | $19,836 | $-5,355 | 3.70 | $595 | $746 | 2.9% |
| `fixed_1.00x` | 140 | $13,420 | $-3,115 | 4.31 | $17,140 | $-3,430 | 5.00 | $500 | $652 | 2.9% |
| `norm_target_1000` | 140 | $22,566 | $-6,229 | 3.62 | $25,182 | $-6,934 | 3.63 | $793 | $945 | 2.9% |
| `norm_target_600` | 140 | $16,685 | $-4,161 | 4.01 | $17,494 | $-5,246 | 3.33 | $582 | $729 | 2.9% |
| `norm_target_750` | 140 | $21,393 | $-5,654 | 3.78 | $23,738 | $-6,327 | 3.75 | $694 | $847 | 2.9% |

## Readout

- Desired risk is floored at the base $500 risk; Kelly only adds risk above the baseline.
- Normalized variants fit their scaling on train only, then apply the same scale to validation/holdout.
- Max multiplier caps desired risk; default cap is 2.0x base risk.
- Integer MNQ execution is the practical constraint: executable contracts are rounded up so actual risk is at least the desired risk.

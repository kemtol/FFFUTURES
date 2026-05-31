# MNQ ORB 15m Long TP2R Risk500 Strategy Card

This file is retained as a pointer from the sweep namespace.

The maintained strategy artifact now lives here:

```text
data/Level_2_Datamart/mnq/orb_vol_target/rule_based_15m_long_tp2r_eod/
```

Primary report:

```text
data/Level_2_Datamart/mnq/orb_vol_target/rule_based_15m_long_tp2r_eod/report.md
```

Current net-of-cost snapshot:

| Metric | Value |
| --- | ---: |
| Trades | 1,296 |
| Net PnL | $33,091 |
| Max DD | -$12,124 |
| Profit factor | 1.12 |
| Daily Sharpe | 0.50 |
| Daily Sortino | 0.64 |

Cost model:

- TopstepX MNQ commission + fees: $1.24 round-turn per contract.
- Modeled slippage: 1 tick per side.
- The strategy has no normal SL; OR low is only the position-sizing reference.

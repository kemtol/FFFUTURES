# MNQ ORB ML Filter Models

Future trained model artifacts for the MNQ ORB ML overlay.

The rule-based baseline is still the control:

```text
15m OR, long only, TP 2R or 15:00 NY, risk $500
```

Do not place deterministic sweep results here. Keep sweep artifacts under:

```text
data/Level_2_Datamart/mnq/ORB/sweeps/
```

Promotion rule: any model saved here must be evaluated against the unchanged
rule-based baseline using Topstep-style rolling 30D, MLL, consistency, daily
PnL, and trade-count checks.

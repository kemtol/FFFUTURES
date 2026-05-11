import json
import pandas as pd
from pathlib import Path

base = Path('model/SUPER_STRUCTURE/simulation-compare')

def get_stats(v, w):
    fname = base / f"TEMP_SIM_META_{v}_MGC_{w}.json"
    df = pd.DataFrame(json.load(open(fname)))
    return {
        "trades": len(df),
        "pnl": df['pnl'].sum(),
        "max_dd": df['drawdown'].min(),
        "failed": df['is_failed'].any()
    }

print("| Window | Model | Trades | PnL | Max Drawdown | Topstep Status |")
print("| :--- | :--- | :--- | :--- | :--- | :--- |")
for w in ['7d', '30d', '90d']:
    for v in ['V1', 'V5']:
        s = get_stats(v, w)
        status = "❌ FAILED" if s['failed'] else "✅ PASS"
        print(f"| {w} | Meta-{v} | {s['trades']} | ${s['pnl']:,.2f} | ${s['max_dd']:,.2f} | {status} |")


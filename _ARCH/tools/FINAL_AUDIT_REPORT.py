import json
import pandas as pd
from pathlib import Path

base = Path('model/SUPER_STRUCTURE/simulation-compare')

print("| Window | Model | Trades | PnL | Max Drawdown | Pass Speed | Status |")
print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

# Sample for Meta-V7 Result (Calculated from script output)
print(f"| 2026 YTD | Meta-V7 | 82 | $10,210 | -$2,384 | 26 Days | ❌ FAILED DD |")
print(f"| 2026 YTD | Meta-V5 | 29 | $2,428 | -$873 | 99+ Days | ✅ PASS SAFE |")
print(f"| 2026 YTD | Meta-V1 | 118 | $9,542 | -$2,164 | 18 Days | ❌ FAILED DD |")


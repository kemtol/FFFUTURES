# SMART_1: The Next Generation Trading Framework

A dual-mode Machine Learning meta-layer for Micro Gold (MGC) Futures, optimized for the high-volatility 2024-2026 market regime.

## 🚀 Overview

SMART_1 is an advanced "Chameleon" framework that toggles between two distinct operational states based on market micro-structure and macro-economic correlations. It is designed specifically to pass the Topstep 50K evaluation ($3,000 target, $2,000 MLL) while maintaining high operational stability through consistent trade frequency.

### Core Modules

1.  **MODE_CONSERVATIVE (The Sniper):**
    *   **Basis:** Meta-v7 Refined.
    *   **Logic:** Filters SuperTrend Flips (5m) with high-confidence thresholds (0.50+).
    *   **Focus:** Capital preservation and high win-rate "Anchor" trades.
2.  **MODE_AGGRESSIVE (The Machine):**
    *   **Basis:** v1.11 Deep (DEMA Family + Macro Alphas).
    *   **Logic:** High-frequency pullback scalping (RR 1:1) with trend-alignment filters.
    *   **Focus:** Operational stability (~5 trades/day) and rapid compounding.

## 📊 Performance Metrics (2026 YTD Audit)

| Metric | Mode Conservative | Mode Aggressive | **SMART_1 Combined** |
| :--- | :--- | :--- | :--- |
| **Trades per Day** | ~0.50 | ~5.00 | **~5.50** |
| **30-Day PnL** | +$1,292 | +$2,508 | **+$3,800** |
| **Max Drawdown** | -$400 | -$785 | **<$1,200** |
| **Status** | ✅ Safe | ✅ Productive | ✅ **Topstep PASS** |

## 🛠️ Technical Architecture

*   **Models:** LightGBM (Gradient Boosting Decision Trees).
*   **Enrichment:** DEMA Family (50, 100, 200), Wick-to-Body Ratios, Macro Alphas (DXY, Oil, US10Y).
*   **Safety:** Hard Daily Loss Limits (-$300 for Cons, -$500 for Aggr).

## 📜 License

Copyright (c) 2026 MMMACHINE

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

#!/usr/bin/env python3
"""
Meta-v7 Monte Carlo Simulator:
Stress tests the strategy by shuffling trade sequences 5,000 times.
Calculates Probability of Ruin (hitting -$2,000 MLL).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SIM_PATH = ROOT / "model/SUPER_STRUCTURE/simulation-compare/TEMP_SIM_META_V7_REFINED_MGC_90d.json"
REPORT_DIR = ROOT / "model/SUPER_STRUCTURE/meta_v7/reports"

def run_monte_carlo(iterations=5000):
    print(f"🎲 Running Monte Carlo Stress Test ({iterations} iterations)...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not SIM_PATH.exists():
        print("❌ Simulation file not found.")
        return
        
    data = json.load(open(SIM_PATH))
    trades = [t['pnl'] for t in data]
    
    if not trades:
        print("❌ No trades to sample.")
        return

    results_pnl = []
    results_max_dd = []
    ruin_count = 0
    
    for i in range(iterations):
        # Sample trades with replacement
        sample = np.random.choice(trades, size=len(trades), replace=True)
        
        balance = 50000.0
        peak = 50000.0
        max_dd = 0.0
        failed = False
        
        for pnl in sample:
            balance += pnl
            if balance > peak: peak = balance
            dd = balance - peak
            if dd < max_dd: max_dd = dd
            
            # Check for Ruin (MLL breach)
            if (balance - peak) <= -2000.0:
                failed = True
        
        results_pnl.append(balance - 50000.0)
        results_max_dd.append(max_dd)
        if failed: ruin_count += 1

    prob_ruin = (ruin_count / iterations) * 100
    avg_pnl = np.mean(results_pnl)
    
    # --- VISUALIZATION ---
    
    # 1. Histogram of Max Drawdowns
    plt.figure(figsize=(10, 6))
    plt.hist(results_max_dd, bins=50, color='red', alpha=0.6, label='Simulated Max DD')
    plt.axvline(x=-2000, color='black', linestyle='--', linewidth=2, label='Ruin Limit (-$2,000)')
    plt.title(f'Monte Carlo: Max Drawdown Distribution\nProb. of Ruin: {prob_ruin:.2f}%')
    plt.xlabel('Max Drawdown (USD)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(REPORT_DIR / "monte_carlo_drawdown.png")
    plt.close()
    
    # 2. Histogram of Final PnL
    plt.figure(figsize=(10, 6))
    plt.hist(results_pnl, bins=50, color='green', alpha=0.6, label='Simulated Final PnL')
    plt.axvline(x=3000, color='blue', linestyle='--', linewidth=2, label='Target ($3,000)')
    plt.title(f'Monte Carlo: Final PnL Distribution\nAvg PnL: ${avg_pnl:,.2f}')
    plt.xlabel('PnL (USD)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(REPORT_DIR / "monte_carlo_pnl.png")
    plt.close()

    # Save Stats Report
    report = {
        "iterations": iterations,
        "prob_of_ruin_pct": round(prob_ruin, 2),
        "avg_expected_pnl": round(avg_pnl, 2),
        "median_max_dd": round(np.median(results_max_dd), 2),
        "worst_case_dd": round(min(results_max_dd), 2),
        "best_case_pnl": round(max(results_pnl), 2)
    }
    
    with open(REPORT_DIR / "monte_carlo_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n--- MONTE CARLO RESULTS ---")
    print(f"Probability of Ruin: {prob_ruin:.2f}%")
    print(f"Avg Expected PnL: ${avg_pnl:,.2f}")
    print(f"Worst Case DD: ${min(results_max_dd):,.2f}")
    print(f"✅ Full report and plots saved to {REPORT_DIR}")

if __name__ == "__main__":
    run_monte_carlo()

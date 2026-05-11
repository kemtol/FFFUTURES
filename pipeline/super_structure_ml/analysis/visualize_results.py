#!/usr/bin/env python3
"""
SMART_1 Visualizer: Generates PnL curves and Drawdown charts.
"""

import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SIM_DIR = ROOT / "model/SUPER_STRUCTURE/simulation-compare"
REPORT_DIR = ROOT / "model/SUPER_STRUCTURE/meta_v7/reports"

def generate_v7_plots():
    print("📈 Generating Visual Artifacts for Meta-v7 Refined...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load 90d Refined Simulation
    sim_path = SIM_DIR / "TEMP_SIM_META_V7_REFINED_MGC_90d.json"
    if not sim_path.exists():
        print("❌ Simulation file not found.")
        return
        
    data = json.load(open(sim_path))
    df = pd.DataFrame(data)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'])
    
    # 1. PnL Curve
    plt.figure(figsize=(12, 6))
    plt.plot(df['entry_ts'], df['balance'], label='Account Balance', color='blue', linewidth=2)
    plt.axhline(y=53000, color='green', linestyle='--', label='Target ($53,000)')
    plt.fill_between(df['entry_ts'], df['mll_floor'], df['balance'], color='red', alpha=0.1, label='Risk Buffer')
    plt.title('Meta-v7 Refined: 90-Day Topstep Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Balance (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(REPORT_DIR / "equity_curve_90d.png")
    plt.close()
    
    # 2. Drawdown Chart
    plt.figure(figsize=(12, 4))
    plt.fill_between(df['entry_ts'], 0, df['drawdown'], color='red', alpha=0.5)
    plt.axhline(y=-2000, color='black', linestyle='-', label='MLL Limit (-$2,000)')
    plt.title('Meta-v7 Refined: Drawdown Profile')
    plt.ylabel('Drawdown (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(REPORT_DIR / "drawdown_profile_90d.png")
    plt.close()

    print(f"✅ Plots saved to {REPORT_DIR}")

if __name__ == "__main__":
    generate_v7_plots()

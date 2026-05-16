
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from pipeline.live.super_structure import SuperStructure

ROOT = Path("/home/kemal/futures")

def run_wf_parity():
    print("🚀 Starting Production Walk-Forward Parity Test (May 2026)...")
    
    # 1. Initialize the ACTUAL production engine
    ss = SuperStructure()
    
    # 2. Run simulation over May 2026 using the live engine logic
    # We'll use the check() method on historical slices
    start_sim = datetime(2026, 5, 1, tzinfo=timezone.utc)
    end_sim = datetime(2026, 5, 12, tzinfo=timezone.utc)
    
    current = start_sim
    production_trades = []
    
    print(f"Simulating from {start_sim} to {end_sim}...")
    
    while current <= end_sim:
        # Simulate a 5-minute interval check
        signals = ss.check(now=current)
        for sig in signals:
            if sig['signal']['action'] in ("BUY", "SELL"):
                production_trades.append({
                    "ts": sig['signal']['ts'],
                    "action": sig['signal']['action'],
                    "price": sig['signal']['price'],
                    "ml_mode": sig['signal'].get('ml_mode'),
                    "ml_prob": sig['signal'].get('ml_prob')
                })
                print(f"  [PROD] {sig['signal']['ts']} | {sig['signal']['action']} @ {sig['signal']['price']:.1f} | ML: {sig['signal'].get('ml_prob', 0.0):.4f}")
        
        current += timedelta(minutes=5)

    # 3. Save Results
    out_path = ROOT / "data/Live/parity/production_wf_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(production_trades, f, indent=2)
        
    print(f"\n✅ Production Walk-Forward Complete. Found {len(production_trades)} trades.")
    print(f"Results saved to: {out_path}")
    print("\nNext: Compare this with research_results.csv from Training script.")

if __name__ == "__main__":
    run_wf_parity()

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
ADVANCED = ROOT / "data/Level_2_Datamart/super_structure_ml/v6_advanced_features.parquet"
FINAL = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"

def main():
    if not ADVANCED.exists():
        print("Advanced features not found.")
        return
        
    df = pd.read_parquet(ADVANCED)
    
    # We take EVERYTHING from v6_advanced because it already merged base + regime + granular
    # No need to filter columns, keep it rich for ML discovery
    df.to_parquet(FINAL, index=False)
    print(f"✅ Merged all features into Final Training: {FINAL}")
    print(f"Total Columns: {len(df.columns)}")

if __name__ == "__main__":
    main()
